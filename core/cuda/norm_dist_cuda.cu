#include "fast_pow.cuh"
#include "norm_dist.h"
#include "forward.cuh"

const int BLOCK_SIZE = 16;
const int BLOCK_CI_SIZE = 8;
const int WARP_SIZE = 32;
const int SUB_BATCH = 32; // for dropout only; must be equal to WARP_SIZE

#define CONST_PTR const float* __restrict__
#define CONST_INDEX_PTR const int* __restrict__
#define PTR float* __restrict__

#define EPS 1e-10f
#define EPS2 1e-6f
#define UNDER_FLOW_EPS 1.175494351e-38f

template<int ip> __device__ __forceinline__
float update_forward(float x, float w, float p) {
    float t = x - w;
    return pow_fun<ip, false>(t, p);
}

template<int ip> __device__ __forceinline__
float update_forward(float x, float w, float p, float r_max_x_sub_w) {
    float t = x - w;
    return pow_fun<ip, false>(t * r_max_x_sub_w, p);
}

__device__ __forceinline__
void normalize(float& output_reg, float ratio, float p) {
    output_reg = output_reg * pow_fun(ratio, -p);
}

// To maximize performance, adjust GROUP_CO, GROUP_B, GROUP_CI and #pragma unroll count for variable t
template <int ip, int GROUP_CI, int GROUP_CO, int GROUP_B, bool has_hw, bool check_ci, bool has_w_ci, bool has_w_co>
__global__
void norm_dist_forward_kernel(CONST_PTR input, CONST_PTR weight,
                              int B, int CO_div_G, int CI_div_G, int HW, int G, PTR output, float p,
                              CONST_INDEX_PTR w_index_ci, CONST_INDEX_PTR w_index_co, int W_CO_div_G, int W_CI_div_G) {
    int b_hw = blockIdx.x * (BLOCK_SIZE * GROUP_B) + (has_hw ? threadIdx.x : threadIdx.y);
    int b[GROUP_B], hw[GROUP_B];
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        int b_hw_i = b_hw + i * BLOCK_SIZE;
        if (has_hw) { b[i] = b_hw_i / HW; hw[i] = b_hw_i % HW; }
        else { b[i] = b_hw_i; hw[i] = 0; }
    }
    int sub_batch = b[0] / SUB_BATCH;
    if (!has_hw) HW = 1;

    if (!has_w_ci) W_CI_div_G = CI_div_G;
    if (!has_w_co) W_CO_div_G = CO_div_G;

    int write_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + (has_hw ? threadIdx.y : threadIdx.x);
    int read_w_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + threadIdx.y;

    __shared__ float blockI[GROUP_CI][GROUP_B][BLOCK_SIZE][BLOCK_SIZE]; // CI * B if has_hw else B * CI
    __shared__ float blockW[GROUP_CI][BLOCK_SIZE][BLOCK_SIZE * GROUP_CO + WARP_SIZE / BLOCK_SIZE]; // CI * CO

    float res[GROUP_B][GROUP_CO];
    float max_output[GROUP_B][GROUP_CO], r_max_output[GROUP_B][GROUP_CO];
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        #pragma unroll
        for (int j = 0; j < GROUP_CO; j++) {
            res[i][j] = 1.0f;
            max_output[i][j] = EPS;
            r_max_output[i][j] = 1.0f / EPS;
        }
    }

    for (int k = 0; k < CI_div_G; k += BLOCK_SIZE * GROUP_CI) {
        #pragma unroll
        for (int kk = 0; kk < GROUP_CI; kk++) {
            #pragma unroll
            for (int i = 0; i < GROUP_B; i++) {
                if (b[i] < B) {
                    int channel = k + kk * BLOCK_SIZE + (has_hw ? threadIdx.y : threadIdx.x);
                    int input_offset = ((b[i] * G + blockIdx.z) * CI_div_G + channel) * HW + hw[i];
                    if (check_ci) blockI[kk][i][threadIdx.y][threadIdx.x] = channel < CI_div_G ? input[input_offset] : 0;
                    else blockI[kk][i][threadIdx.y][threadIdx.x] = input[input_offset];
                }
            }
        }
        #pragma unroll
        for (int kk = 0; kk < GROUP_CI; kk++) {
            #pragma unroll
            for (int i = 0; i < GROUP_CO; i++) {
                int out_channel = read_w_co + i * BLOCK_SIZE;
                if (out_channel < CO_div_G) {
                    int channel = k + kk * BLOCK_SIZE + threadIdx.x;
                    int in_channel = has_w_ci && channel < CI_div_G ?
                        w_index_ci[(sub_batch * G + blockIdx.z) * CI_div_G + channel] : channel;
                    if (has_w_co) out_channel = w_index_co[(sub_batch * G + blockIdx.z) * CO_div_G + out_channel];
                    int weight_offset = (blockIdx.z * W_CO_div_G + out_channel) * W_CI_div_G + in_channel;
                    if (check_ci) blockW[kk][threadIdx.x][threadIdx.y + i * BLOCK_SIZE] =
                        channel < CI_div_G ? weight[weight_offset] : 0;
                    else blockW[kk][threadIdx.x][threadIdx.y + i * BLOCK_SIZE] = weight[weight_offset];
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < GROUP_CI; kk++) {
            #pragma unroll(4)
            for (int t = 0; t < BLOCK_SIZE; t++) {
                #pragma unroll
                for (int i = 0; i < GROUP_B; i++) {
                    #pragma unroll
                    for (int j = 0; j < GROUP_CO; j++) {
                        float x = has_hw ? blockI[kk][i][t][threadIdx.x] : blockI[kk][i][threadIdx.y][t];
                        float w = blockW[kk][t][(has_hw ? threadIdx.y : threadIdx.x) + j * BLOCK_SIZE];
                        max_output[i][j] = max(max_output[i][j], abs(x - w));
                    }
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < GROUP_B; i++) {
            #pragma unroll
            for (int j = 0; j < GROUP_CO; j++) {
                float ratio = max_output[i][j] * r_max_output[i][j];
                if (ratio > 1.0f + EPS2) {
                    normalize(res[i][j], ratio, p);
                    r_max_output[i][j] = __frcp_rn(max_output[i][j]);
                }
            }
        }
        #pragma unroll
        for (int kk = 0; kk < GROUP_CI; kk++) {
            #pragma unroll
            for (int t = 0; t < BLOCK_SIZE; t++) {
                #pragma unroll
                for (int i = 0; i < GROUP_B; i++) {
                    #pragma unroll
                    for (int j = 0; j < GROUP_CO; j++) {
                        float x = has_hw ? blockI[kk][i][t][threadIdx.x] : blockI[kk][i][threadIdx.y][t];
                        float w = blockW[kk][t][(has_hw ? threadIdx.y : threadIdx.x) + j * BLOCK_SIZE];
                        res[i][j] += update_forward<ip>(x, w, p, r_max_output[i][j]);
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        if (b[i] < B) {
            #pragma unroll
            for (int j = 0; j < GROUP_CO; j++) {
                int channel = write_co + j * BLOCK_SIZE;
                if (channel < CO_div_G) {
                    float ans = pow_fun(res[i][j], __frcp_rn(p)) * max_output[i][j];
                    output[((b[i] * G + blockIdx.z) * CO_div_G + channel) * HW + hw[i]] = ans;
                }
            }
        }
    }
}

// To maximize performance, adjust GROUP_CO, GROUP_B, GROUP_CI and #pragma unroll count for variable t
template <int ip, int GROUP_CO, int GROUP_B, bool has_hw, bool has_max, bool check_ci, bool has_w_ci, bool has_w_co>
__global__
void norm_dist_forward_kernel(CONST_PTR input, CONST_PTR weight, CONST_PTR max_output,
                              int B, int CO_div_G, int CI_div_G, int HW, int G, PTR output, float p,
                              CONST_INDEX_PTR w_index_ci, CONST_INDEX_PTR w_index_co, int W_CO_div_G, int W_CI_div_G) {
    int b_hw = blockIdx.x * (BLOCK_SIZE * GROUP_B) + (has_hw ? threadIdx.x : threadIdx.y);
    int b[GROUP_B], hw[GROUP_B];
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        int b_hw_i = b_hw + i * BLOCK_SIZE;
        if (has_hw) { b[i] = b_hw_i / HW; hw[i] = b_hw_i % HW; }
        else { b[i] = b_hw_i; hw[i] = 0; }
    }
    int sub_batch = b[0] / SUB_BATCH;
    if (!has_hw) HW = 1;

    if (!has_w_ci) W_CI_div_G = CI_div_G;
    if (!has_w_co) W_CO_div_G = CO_div_G;

    int write_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + (has_hw ? threadIdx.y : threadIdx.x);
    int read_w_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + threadIdx.y;

    __shared__ float blockI[GROUP_B][BLOCK_SIZE][BLOCK_SIZE]; // CI * B if has_hw else B * CI
    __shared__ float blockW[BLOCK_SIZE][BLOCK_SIZE * GROUP_CO + WARP_SIZE / BLOCK_SIZE]; // CI * CO

    float res[GROUP_B][GROUP_CO];
    float r_max_output[GROUP_B][GROUP_CO];
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        #pragma unroll
        for (int j = 0; j < GROUP_CO; j++) {
            if (has_max) {
                int channel = write_co + j * BLOCK_SIZE;
                if (b[i] < B && channel < CO_div_G)
                    r_max_output[i][j] = __frcp_rn(max_output[((b[i] * G + blockIdx.z) * CO_div_G + channel) * HW + hw[i]]);
                res[i][j] = pow_fun<ip, true>(EPS * r_max_output[i][j], p);
            }
            else res[i][j] = 0;
        }
    }

    for (int k = 0; k < CI_div_G; k += BLOCK_SIZE) {
        #pragma unroll
        for (int i = 0; i < GROUP_B; i++) {
            if (b[i] < B) {
                int channel = k + (has_hw ? threadIdx.y : threadIdx.x);
                int input_offset = ((b[i] * G + blockIdx.z) * CI_div_G + channel) * HW + hw[i];
                if (check_ci) blockI[i][threadIdx.y][threadIdx.x] = channel < CI_div_G ? input[input_offset] : 0;
                else blockI[i][threadIdx.y][threadIdx.x] = input[input_offset];
            }
        }
        #pragma unroll
        for (int i = 0; i < GROUP_CO; i++) {
            int out_channel = read_w_co + i * BLOCK_SIZE;
            if (out_channel < CO_div_G) {
                int channel = k + threadIdx.x;
                int in_channel = has_w_ci && channel < CI_div_G ?
                    w_index_ci[(sub_batch * G + blockIdx.z) * CI_div_G + channel] : channel;
                if (has_w_co) out_channel = w_index_co[(sub_batch * G + blockIdx.z) * CO_div_G + out_channel];
                int weight_offset = (blockIdx.z * W_CO_div_G + out_channel) * W_CI_div_G + in_channel;
                if (check_ci) blockW[threadIdx.x][threadIdx.y + i * BLOCK_SIZE] =
                    channel < CI_div_G ? weight[weight_offset] : 0;
                else blockW[threadIdx.x][threadIdx.y + i * BLOCK_SIZE] = weight[weight_offset];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < BLOCK_SIZE; t++) {
            #pragma unroll
            for (int i = 0; i < GROUP_B; i++) {
                #pragma unroll
                for (int j = 0; j < GROUP_CO; j++) {
                    float x = has_hw ? blockI[i][t][threadIdx.x] : blockI[i][threadIdx.y][t];
                    float w = blockW[t][(has_hw ? threadIdx.y : threadIdx.x) + j * BLOCK_SIZE];
                    if (has_max) res[i][j] += update_forward<ip>(x, w, p, r_max_output[i][j]);
                    else res[i][j] += update_forward<ip>(x, w, p);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        if (b[i] < B) {
            #pragma unroll
            for (int j = 0; j < GROUP_CO; j++) {
                int channel = write_co + j * BLOCK_SIZE;
                if (channel < CO_div_G) {
                    float ans;
                    if (has_max) ans = __fdiv_rn(pow_fun(res[i][j], __frcp_rn(p)), r_max_output[i][j]);
                    else ans = pow_fun(max(res[i][j], UNDER_FLOW_EPS), __frcp_rn(p)); // for underflow case
                    output[((b[i] * G + blockIdx.z) * CO_div_G + channel) * HW + hw[i]] = ans;
                }
            }
        }
    }
}

template <int ip> __device__ __forceinline__
float update_backward_input(float x, float w, float r_o, float g, float p) {
    float t = x - w;
    if ((ip & 1) && ip >= 0) return g * pow_fun<ip, true>(t * r_o, p);
    return g * pow_fun<ip, false>(t * r_o, p) * (t >= 0 ? 1 : -1);
}

// To maximize performance, adjust GROUP_B, GROUP_CI and #pragma unroll count for variable t
template <int ip, int GROUP_CI, int GROUP_B, bool has_hw, bool check_co, bool has_w_ci, bool has_w_co> __global__
void norm_dist_backward_input_kernel(CONST_PTR grad_output, CONST_PTR input, CONST_PTR weight, CONST_PTR output,
                                     int B, int CO_div_G, int CI_div_G, int HW, int G, PTR grad_input, float p,
                                     CONST_INDEX_PTR w_index_ci, CONST_INDEX_PTR w_index_co, int W_CO_div_G,
                                     int W_CI_div_G) {
    int b_hw = blockIdx.x * (BLOCK_SIZE * GROUP_B) + (has_hw ? threadIdx.x : threadIdx.y);
    int b[GROUP_B], hw[GROUP_B];
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        int b_hw_i = b_hw + i * BLOCK_SIZE;
        if (has_hw) { b[i] = b_hw_i / HW; hw[i] = b_hw_i % HW; }
        else { b[i] = b_hw_i; hw[i] = 0; }
    }
    int sub_batch = b[0] / SUB_BATCH;
    if (!has_hw) HW = 1;

    if (!has_w_ci) W_CI_div_G = CI_div_G;
    if (!has_w_co) W_CO_div_G = CO_div_G;

    int write_ci = blockIdx.y * (BLOCK_SIZE * GROUP_CI) + (has_hw ? threadIdx.y : threadIdx.x);
    int read_w_ci = blockIdx.y * (BLOCK_SIZE * GROUP_CI) + threadIdx.x;

    __shared__ float blockO[GROUP_B][BLOCK_SIZE][BLOCK_SIZE]; // CO * B if has_hw else B * CO
    __shared__ float blockG[GROUP_B][BLOCK_SIZE][BLOCK_SIZE]; // CO * B if has_hw else B * CO
    __shared__ float blockW[GROUP_CI][BLOCK_SIZE][BLOCK_SIZE]; // CO * CI

    p -= 1;

    float res[GROUP_B][GROUP_CI], x[GROUP_B][GROUP_CI];
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        #pragma unroll
        for (int j = 0; j < GROUP_CI; j++) {
            res[i][j] = 0;
            if (b[i] < B && write_ci + j * BLOCK_SIZE < CI_div_G)
                x[i][j] = input[((b[i] * G + blockIdx.z) * CI_div_G + write_ci + j * BLOCK_SIZE) * HW + hw[i]];
            else x[i][j] = 0;
        }
    }

    for (int k = 0; k < CO_div_G; k += BLOCK_SIZE) {
        #pragma unroll
        for (int i = 0; i < GROUP_B; i++) {
            if (b[i] < B) {
                int channel = k + (has_hw ? threadIdx.y : threadIdx.x);
                int output_offset = ((b[i] * G + blockIdx.z) * CO_div_G + channel) * HW + hw[i];
                if (check_co) {
                    blockO[i][threadIdx.y][threadIdx.x] = channel < CO_div_G ? __frcp_rn(output[output_offset]) : 0;
                    blockG[i][threadIdx.y][threadIdx.x] = channel < CO_div_G ? grad_output[output_offset] : 0;
                }
                else {
                    blockO[i][threadIdx.y][threadIdx.x] = __frcp_rn(output[output_offset]);
                    blockG[i][threadIdx.y][threadIdx.x] = grad_output[output_offset];
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < GROUP_CI; i++) {
            int in_channel = read_w_ci + i * BLOCK_SIZE;
            if (in_channel < CI_div_G) {
                int channel = k + threadIdx.y;
                int out_channel = has_w_co && channel < CO_div_G ?
                    w_index_co[(sub_batch * G + blockIdx.z) * CO_div_G + channel] : channel;
                if (has_w_ci) in_channel = w_index_ci[(sub_batch * G + blockIdx.z) * CI_div_G + in_channel];
                int weight_offset = (blockIdx.z * W_CO_div_G + out_channel) * W_CI_div_G + in_channel;
                if (check_co) blockW[i][threadIdx.y][threadIdx.x] =
                    channel < CO_div_G ? weight[weight_offset] : 0;
                else blockW[i][threadIdx.y][threadIdx.x] = weight[weight_offset];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < BLOCK_SIZE; t++) {
            #pragma unroll
            for (int i = 0; i < GROUP_B; i++) {
                #pragma unroll
                for (int j = 0; j < GROUP_CI; j++) {
                    float r_o = has_hw ? blockO[i][t][threadIdx.x] : blockO[i][threadIdx.y][t];
                    float g = has_hw ? blockG[i][t][threadIdx.x] : blockG[i][threadIdx.y][t];
                    float w = blockW[j][t][has_hw ? threadIdx.y : threadIdx.x];
                    res[i][j] += update_backward_input<ip>(x[i][j], w, r_o, g, p);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        if (b[i] < B) {
            #pragma unroll
            for (int j = 0; j < GROUP_CI; j++) {
                int channel = write_ci + j * BLOCK_SIZE;
                if (channel < CI_div_G) {
                    grad_input[((b[i] * G + blockIdx.z) * CI_div_G + channel) * HW + hw[i]] = res[i][j];
                }
            }
        }
    }
}

// To maximize performance, adjust GROUP_CI, GROUP_B and #pragma unroll count for variable t
template <int ip, int GROUP_CI, int GROUP_B, bool has_hw, bool has_w_ci, bool has_w_co> __global__
void norm_dist_backward_input_weight_kernel(CONST_PTR grad_output, CONST_PTR input, CONST_PTR weight, CONST_PTR output,
                                            int B, int CO_div_G, int CI_div_G, int HW, int G,
                                            PTR grad_input, PTR grad_weight, float p, CONST_INDEX_PTR w_index_ci,
                                            CONST_INDEX_PTR w_index_co, int W_CO_div_G, int W_CI_div_G) {
    int threadIdx_low = threadIdx.x & (BLOCK_CI_SIZE - 1);
    int threadIdx_high = threadIdx.y * (WARP_SIZE / BLOCK_CI_SIZE) | (threadIdx.x / BLOCK_CI_SIZE);

    int b_hw = blockIdx.x * WARP_SIZE + (has_hw ? threadIdx.x : threadIdx_high), b, hw;
    if (has_hw) { b = b_hw / HW; hw = b_hw % HW; }
    else { b = b_hw; hw = 0; }
    int write_ci = blockIdx.y * (BLOCK_CI_SIZE * GROUP_CI) +  (has_hw ? threadIdx.y : threadIdx_low);
    int read_w_ci = blockIdx.y * (BLOCK_CI_SIZE * GROUP_CI) + threadIdx_low;
    int sub_batch = b / SUB_BATCH;
    if (!has_hw) HW = 1;

    if (!has_w_ci) W_CI_div_G = CI_div_G;
    if (!has_w_co) W_CO_div_G = CO_div_G;

    __shared__ float blockO[WARP_SIZE][WARP_SIZE]; // has_hw ? CO * B : B * CO
    __shared__ float blockGO[WARP_SIZE][WARP_SIZE]; // has_hw ? CO * B : B * CO
    __shared__ float blockW[GROUP_CI][BLOCK_CI_SIZE][WARP_SIZE + WARP_SIZE / BLOCK_CI_SIZE]; // CI * CO
    __shared__ float blockI[GROUP_CI][BLOCK_CI_SIZE][WARP_SIZE + WARP_SIZE / BLOCK_CI_SIZE]; // CI * B

    p -= 1;

    #pragma unroll
    for (int j = 0; j < GROUP_CI; j++) {
        int offset = ((b * G + blockIdx.z) * CI_div_G + write_ci + j * BLOCK_CI_SIZE) * HW + hw;
        float tmp = b < B && write_ci + j * BLOCK_CI_SIZE < CI_div_G ? input[offset] : 0;
        if (has_hw) blockI[j][threadIdx.y][threadIdx.x] = tmp;
        else blockI[j][threadIdx_low][threadIdx_high] = tmp;
    }

    float grad_x[GROUP_B][GROUP_CI];
    #pragma unroll
    for (int s = 0; s < GROUP_B; s++) {
        #pragma unroll
        for (int i = 0; i < GROUP_CI; i++)
            grad_x[s][i] = 0;
    }

    for (int k = 0; k < CO_div_G; k += WARP_SIZE) {
        #pragma unroll
        for (int i = 0; i < WARP_SIZE; i += BLOCK_CI_SIZE) {
            int read_output_co = k + (has_hw ? threadIdx.y + i: threadIdx.x);
            int read_output_b_hw = blockIdx.x * WARP_SIZE + (has_hw ? threadIdx.x : threadIdx.y + i);
            int read_b = has_hw ? read_output_b_hw / HW : read_output_b_hw;
            int read_hw = has_hw ? read_output_b_hw % HW : 0;
            int offset = ((read_b * G + blockIdx.z) * CO_div_G + read_output_co) * HW + read_hw;
            if (read_b < B && read_output_co < CO_div_G) {
                blockO[threadIdx.y + i][threadIdx.x] = __frcp_rn(output[offset]);
                blockGO[threadIdx.y + i][threadIdx.x] = grad_output[offset];
            }
            else blockO[threadIdx.y + i][threadIdx.x] = blockGO[threadIdx.y + i][threadIdx.x] = 0;
        }
        #pragma unroll
        for (int i = 0; i < GROUP_CI; i++) {
            int in_channel = read_w_ci + i * BLOCK_CI_SIZE;
            int channel = k + threadIdx_high;
            if (in_channel < CI_div_G) {
                int out_channel = has_w_co && channel < CO_div_G ?
                    w_index_co[(sub_batch * G + blockIdx.z) * CO_div_G + channel] : channel;
                if (has_w_ci) in_channel = w_index_ci[(sub_batch * G + blockIdx.z) * CI_div_G + in_channel];
                int weight_offset = (blockIdx.z * W_CO_div_G + out_channel) * W_CI_div_G + in_channel;
                blockW[i][threadIdx_low][threadIdx_high] = channel < CO_div_G ? weight[weight_offset] : 0;
            }
        }
        __syncthreads();
        float grad_w[GROUP_CI];
        #pragma unroll
        for (int i = 0; i < GROUP_CI; i++)
            grad_w[i] = 0;
        #pragma unroll(4)
        for (int t = 0; t < (WARP_SIZE / GROUP_B); t++) {
            #pragma unroll
            for (int i = 0; i < GROUP_CI; i++) {
                float sum_res;
                #pragma unroll
                for (int s = 0; s < GROUP_B; s++) {
                    int b = threadIdx.x ^ (WARP_SIZE / GROUP_B * s);
                    int co = threadIdx.x ^ t;
                    float w = blockW[i][threadIdx.y][co];
                    float x = blockI[i][threadIdx.y][b];
                    float ro = has_hw ? blockO[co][b] : blockO[b][co];
                    float g = has_hw ? blockGO[co][b] : blockGO[b][co];
                    float res = update_backward_input<ip>(x, w, ro, g, p);
                    if (s == 0) sum_res = res;
                    else sum_res += res;
                    grad_x[s][i] += res;
                }
                grad_w[i] -= __shfl_xor_sync(0xffffffff, sum_res, t); // grad at co=threadIdx.x
            }
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < GROUP_CI; i++)
            blockW[i][threadIdx.y][threadIdx.x] = grad_w[i];
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < GROUP_CI; i++) {
            int in_channel = read_w_ci + i * BLOCK_CI_SIZE;
            int out_channel = k + threadIdx_high;
            if (in_channel < CI_div_G && out_channel < CO_div_G) {
                if (has_w_ci) in_channel = w_index_ci[(sub_batch * G + blockIdx.z) * CI_div_G + in_channel];
                if (has_w_co) out_channel = w_index_co[(sub_batch * G + blockIdx.z) * CO_div_G + out_channel];
                int weight_offset = (blockIdx.z * W_CO_div_G + out_channel) * W_CI_div_G + in_channel;
                atomicAdd(&grad_weight[weight_offset], blockW[i][threadIdx_low][threadIdx_high]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int s = 1; s < GROUP_B; s++) {
        #pragma unroll
        for (int i = 0; i < GROUP_CI; i++)
            grad_x[0][i] += __shfl_xor_sync(0xffffffff, grad_x[s][i], WARP_SIZE / GROUP_B * s);
    }
    if (!has_hw) {
        #pragma unroll
        for (int i = 0; i < GROUP_CI; i++)
            blockI[i][threadIdx.y][threadIdx.x] = grad_x[0][i];
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < GROUP_CI; i++)
            grad_x[0][i] = blockI[i][threadIdx_low][threadIdx_high];
    }
    if (b < B) {
        #pragma unroll
        for (int i = 0; i < GROUP_CI; i++) {
            int offset = ((b * G + blockIdx.z) * CI_div_G + write_ci + i * BLOCK_CI_SIZE) * HW + hw;
            if (write_ci + i * BLOCK_CI_SIZE < CI_div_G) grad_input[offset] = grad_x[0][i];
        }
    }
}


create_helper(forward, norm_dist)
create_helper(backward_input, norm_dist)
create_helper(backward_input_weight, norm_dist)

template <int ip>
void norm_dist<ip>::forward(const float* input, const float* weight,
                            int B, int CO, int CI, int G, int HW, float* output, float p,
                            const int* w_index_ci, const int* w_index_co, int W_CO, int W_CI) {
    const int GROUP_CI = 2;
    const int GROUP_CO = 4;
    const int GROUP_B = 2;
    int CI_div_G = CI / G, CO_div_G = CO / G;
    bool has_hw = HW > 1;
    bool check_ci = CI_div_G % (GROUP_CI * BLOCK_SIZE) != 0;
    bool has_w_ci = w_index_ci != nullptr;
    bool has_w_co = w_index_co != nullptr;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B * HW - 1) / (BLOCK_SIZE * GROUP_B) + 1, (CO_div_G - 1) / (BLOCK_SIZE * GROUP_CO) + 1, G);
    auto tuple = std::make_tuple(has_hw, check_ci, has_w_ci, has_w_co);
    static_assert(SUB_BATCH % (GROUP_B * BLOCK_SIZE) == 0, "static assertion failed");
    Call<decltype(tuple)>::call<forward_helper>(std::integer_sequence<int, ip, GROUP_CI, GROUP_CO, GROUP_B>{}, tuple,
        dimGrid, dimBlock,
        input, weight, B, CO_div_G, CI_div_G, HW, G, output, p, w_index_ci, w_index_co, W_CO / G, W_CI / G);
}

template <int ip>
void norm_dist<ip>::forward_with_max(const float* input, const float* weight, const float* max_output,
                                     int B, int CO, int CI, int G, int HW, float* output, float p,
                                     const int* w_index_ci, const int* w_index_co, int W_CO, int W_CI) {
    const int GROUP_CO = 4;
    const int GROUP_B = 2;
    int CI_div_G = CI / G, CO_div_G = CO / G;
    bool has_hw = HW > 1;
    bool has_max = max_output != nullptr;
    bool check_ci = CI_div_G % BLOCK_SIZE != 0;
    bool has_w_ci = w_index_ci != nullptr;
    bool has_w_co = w_index_co != nullptr;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B * HW - 1) / (BLOCK_SIZE * GROUP_B) + 1, (CO_div_G - 1) / (BLOCK_SIZE * GROUP_CO) + 1, G);
    auto tuple = std::make_tuple(has_hw, has_max, check_ci, has_w_ci, has_w_co);
    static_assert(SUB_BATCH % (GROUP_B * BLOCK_SIZE) == 0, "static assertion failed");
    Call<decltype(tuple)>::call<forward_helper>(std::integer_sequence<int, ip, GROUP_CO, GROUP_B>{}, tuple,
        dimGrid, dimBlock,
        input, weight, max_output, B, CO_div_G, CI_div_G, HW, G, output, p, w_index_ci, w_index_co, W_CO / G, W_CI / G);
}

template <int ip>
void norm_dist<ip>::backward_input(const float* grad_output, const float* input, const float* weight, const float* output,
                                   int B, int CO, int CI, int G, int HW, float* grad_input, float p,
                                   const int* w_index_ci, const int* w_index_co, int W_CO, int W_CI) {
    const int GROUP_CI = 4;
    const int GROUP_B = 2;
    int CI_div_G = CI / G, CO_div_G = CO / G;
    bool has_hw = HW > 1;
    bool check_co = CO_div_G % BLOCK_SIZE != 0;
    bool has_w_ci = w_index_ci != nullptr;
    bool has_w_co = w_index_co != nullptr;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B * HW - 1) / (BLOCK_SIZE * GROUP_B) + 1, (CI_div_G - 1) / (BLOCK_SIZE * GROUP_CI) + 1, G);
    auto tuple = std::make_tuple(has_hw, check_co, has_w_ci, has_w_co);
    static_assert(SUB_BATCH % (GROUP_B * BLOCK_SIZE) == 0, "static assertion failed");
    Call<decltype(tuple)>::call<backward_input_helper>(std::integer_sequence<int, ip, GROUP_CI, GROUP_B>{}, tuple,
        dimGrid, dimBlock,
        grad_output, input, weight, output, B, CO_div_G, CI_div_G, HW, G, grad_input, p,
        w_index_ci, w_index_co, W_CO / G, W_CI / G);
}

template <int ip>
void norm_dist<ip>::backward_input_weight(const float* grad_output, const float* input, const float* weight,
                                          const float* output, int B, int CO, int CI, int G, int HW,
                                          float* grad_input, float* grad_weight, float p,
                                          const int* w_index_ci, const int* w_index_co, int W_CO, int W_CI) {
    const int GROUP_CI = 6;
    const int GROUP_B = 2; // to reduce shfl_xor
    int CI_div_G = CI / G, CO_div_G = CO / G;
    bool has_hw = HW > 1;
    bool has_w_ci = w_index_ci != nullptr;
    bool has_w_co = w_index_co != nullptr;
    dim3 dimBlock(WARP_SIZE, BLOCK_CI_SIZE);
    dim3 dimGrid((B * HW - 1) / WARP_SIZE + 1, (CI_div_G - 1) / (BLOCK_CI_SIZE * GROUP_CI) + 1, G);
    auto tuple = std::make_tuple(has_hw, has_w_ci, has_w_co);
    static_assert(SUB_BATCH % WARP_SIZE == 0, "static assertion failed");
    cudaMemset(grad_weight, 0, W_CO * (W_CI / G) * sizeof(float));
    Call<decltype(tuple)>::call<backward_input_weight_helper>(std::integer_sequence<int, ip, GROUP_CI, GROUP_B>{}, tuple,
        dimGrid, dimBlock,
        grad_output, input, weight, output, B, CO_div_G, CI_div_G, HW, G, grad_input, grad_weight, p,
        w_index_ci, w_index_co, W_CO / G, W_CI / G);
}

#define build_p(ip) \
    template struct norm_dist<ip>;

build_p(0)
build_p(1)
build_p(2)
build_p(3)
build_p(4)
build_p(5)
build_p(6)
build_p(7)
build_p(8)
build_p(-1)

