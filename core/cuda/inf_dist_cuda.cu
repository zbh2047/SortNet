#include "norm_dist.h"
#include "forward.cuh"

const int BLOCK_SIZE = 16;
const int WARP_SIZE = 32;
const int MAX_BLOCK_CO_SIZE = 32;
const int MAX_BLOCK_B_SIZE = 16;
const int SUB_BATCH = 32; // for dropout only; must be equal to WARP_SIZE

#define CONST_PTR const float* __restrict__
#define CONST_INDEX_PTR const int* __restrict__
#define PTR float* __restrict__

#define EPS 1e-10f

// To maximize performance, adjust GROUP_CO, GROUP_B, GROUP_CI and #pragma unroll count for variable t
template <int GROUP_CI, int GROUP_CO, int GROUP_B, bool has_hw, bool check_ci, bool ci_split> __global__
void inf_dist_forward_kernel(CONST_PTR input, CONST_PTR weight,
                             int B, int CO_div_G, int CI_div_G, int HW, int G, PTR output) {
    int b_hw = blockIdx.x * (BLOCK_SIZE * GROUP_B) + (has_hw ? threadIdx.x : threadIdx.y);
    int b[GROUP_B], hw[GROUP_B];
    int g = blockIdx.z % G, ci_split_id = blockIdx.z / G;
    int start_ci = ci_split_id * (CI_div_G / WARP_SIZE) / (gridDim.z / G) * WARP_SIZE;
    int end_ci = (ci_split_id + 1) * (CI_div_G / WARP_SIZE) / (gridDim.z / G) * WARP_SIZE;
    if (blockIdx.z / G == gridDim.z / G - 1) end_ci = CI_div_G;
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        int b_hw_i = b_hw + i * BLOCK_SIZE;
        if (has_hw) { b[i] = b_hw_i / HW; hw[i] = b_hw_i % HW; }
        else { b[i] = b_hw_i; hw[i] = 0; }
    }
    if (!has_hw) HW = 1;

    int write_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + (has_hw ? threadIdx.y : threadIdx.x);
    int read_w_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + threadIdx.y;

    __shared__ float blockI[GROUP_CI][GROUP_B][BLOCK_SIZE][BLOCK_SIZE]; // CI * B if has_hw else B * CI
    __shared__ float blockW[GROUP_CI][BLOCK_SIZE][BLOCK_SIZE * GROUP_CO + WARP_SIZE / BLOCK_SIZE]; // CI * CO

    float max_output[GROUP_B][GROUP_CO];
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        #pragma unroll
        for (int j = 0; j < GROUP_CO; j++)
            max_output[i][j] = EPS;
    }

    for (int k = start_ci; k < end_ci; k += BLOCK_SIZE * GROUP_CI) {
        #pragma unroll
        for (int kk = 0; kk < GROUP_CI; kk++) {
            #pragma unroll
            for (int i = 0; i < GROUP_B; i++) {
                if (b[i] < B) {
                    int channel = k + kk * BLOCK_SIZE + (has_hw ? threadIdx.y : threadIdx.x);
                    int input_offset = ((b[i] * G + g) * CI_div_G + channel) * HW + hw[i];
                    if (check_ci) blockI[kk][i][threadIdx.y][threadIdx.x] = channel < end_ci ? input[input_offset] : 0;
                    else blockI[kk][i][threadIdx.y][threadIdx.x] = input[input_offset];
                }
            }
        }
        #pragma unroll
        for (int kk = 0; kk < GROUP_CI; kk++) {
            #pragma unroll
            for (int i = 0; i < GROUP_CO; i++) {
                if (read_w_co + i * BLOCK_SIZE < CO_div_G) {
                    int channel = k + kk * BLOCK_SIZE + threadIdx.x;
                    int weight_offset = (g * CO_div_G + read_w_co + i * BLOCK_SIZE) * CI_div_G + channel;
                    if (check_ci) blockW[kk][threadIdx.x][threadIdx.y + i * BLOCK_SIZE] =
                        channel < end_ci ? weight[weight_offset] : 0;
                    else blockW[kk][threadIdx.x][threadIdx.y + i * BLOCK_SIZE] = weight[weight_offset];
                }
            }
        }
        __syncthreads();
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
                        max_output[i][j] = max(max_output[i][j], abs(x - w));
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
                    int offset = ((b[i] * G + g) * CO_div_G + channel) * HW + hw[i];
                    if (ci_split) atomicMax((int*)&output[offset], __float_as_int(max_output[i][j]));
                     // note that the result is always non-negative so such conversion is correct
                    else output[offset] = max_output[i][j];
                }
            }
        }
    }
}

// To maximize performance, adjust GROUP_CO, GROUP_B, GROUP_CI and #pragma unroll count for variable t
template <int GROUP_CI, int GROUP_CO, int GROUP_B, bool has_hw, bool check_ci, bool has_w_ci, bool has_w_co> __global__
void inf_dist_forward_kernel(CONST_PTR input, CONST_PTR weight,
                             int B, int CO_div_G, int CI_div_G, int HW, int G, PTR output, int* __restrict__ pos,
                             CONST_INDEX_PTR w_index_ci, CONST_INDEX_PTR w_index_co, int W_CO_div_G, int W_CI_div_G) {
    int b_hw = blockIdx.x * (BLOCK_SIZE * GROUP_B) + (has_hw ? threadIdx.x : threadIdx.y);
    int b[GROUP_B], hw[GROUP_B];
    int g = blockIdx.z;
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

    float max_output[GROUP_B][GROUP_CO];
    int res_pos[GROUP_B][GROUP_CO];
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        #pragma unroll
        for (int j = 0; j < GROUP_CO; j++) {
            max_output[i][j] = EPS;
            res_pos[i][j] = 0;
        }
    }

    for (int k = 0; k < CI_div_G; k += BLOCK_SIZE * GROUP_CI) {
        #pragma unroll
        for (int kk = 0; kk < GROUP_CI; kk++) {
            #pragma unroll
            for (int i = 0; i < GROUP_B; i++) {
                if (b[i] < B) {
                    int channel = k + kk * BLOCK_SIZE + (has_hw ? threadIdx.y : threadIdx.x);
                    int input_offset = ((b[i] * G + g) * CI_div_G + channel) * HW + hw[i];
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
                        w_index_ci[(sub_batch * G + g) * CI_div_G + channel] : channel;
                    if (has_w_co) out_channel = w_index_co[(sub_batch * G + g) * CO_div_G + out_channel];
                    int weight_offset = (g * W_CO_div_G + out_channel) * W_CI_div_G + in_channel;
                    if (check_ci) blockW[kk][threadIdx.x][threadIdx.y + i * BLOCK_SIZE] =
                        channel < CI_div_G ? weight[weight_offset] : 0;
                    else blockW[kk][threadIdx.x][threadIdx.y + i * BLOCK_SIZE] = weight[weight_offset];
                }
            }
        }
        __syncthreads();
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
                        float tmp1 = x - w, tmp2 = abs(tmp1);
                        if (tmp2 > max_output[i][j]) {
                            max_output[i][j] = tmp2;
                            res_pos[i][j] = k + kk * BLOCK_SIZE + t + (tmp1 >= 0 ? 0 : 1 << 31);
                        }
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
                    int offset = ((b[i] * G + g) * CO_div_G + channel) * HW + hw[i];
                    output[offset] = max_output[i][j];
                    pos[offset] = res_pos[i][j];
                }
            }
        }
    }
}

template <int BLOCK_CO_SIZE, int BLOCK_B_SIZE, bool has_hw>
__global__ void inf_dist_backward_input_kernel(CONST_PTR grad_output, const int* __restrict__ pos,
                                                int B, int CO_div_G, int CI_div_G, int HW, int G, PTR grad_input) {
    if (!has_hw) HW = 1;
    #pragma unroll
    for (int j = 0; j < BLOCK_B_SIZE; j += BLOCK_SIZE){
        int b_hw = blockIdx.x * BLOCK_B_SIZE + j + (has_hw ? threadIdx.x : threadIdx.y);
        int b = has_hw ? b_hw / HW : b_hw;
        int hw = has_hw ? b_hw % HW : 0;
        int co = blockIdx.y * BLOCK_CO_SIZE + (has_hw ? threadIdx.y : threadIdx.x);
        int offset = ((b * G + blockIdx.z) * CO_div_G + co) * HW + hw;
        #pragma unroll
        for (int i = 0; i < BLOCK_CO_SIZE; i += BLOCK_SIZE){
            if (b < B && co + i < CO_div_G) {
                int pos_reg = pos[offset + i * HW];
                float grad = grad_output[offset + i * HW];
                int index = pos_reg & (~(1 << 31));
                float value = pos_reg >= 0 ? grad : -grad;
                atomicAdd(&grad_input[((b * G + blockIdx.z) * CI_div_G + index) * HW + hw], value);
            }
        }
    }
}

template <int BLOCK_CO_SIZE, int BLOCK_B_SIZE, bool has_hw, bool has_w_ci, bool has_w_co>
__global__ void inf_dist_backward_input_weight_kernel(CONST_PTR grad_output, const int* __restrict__ pos,
                                                      int B, int CO_div_G, int CI_div_G, int HW, int G,
                                                      PTR grad_input, PTR grad_weight,
                                                      CONST_INDEX_PTR w_index_ci, CONST_INDEX_PTR w_index_co,
                                                      int W_CO_div_G, int W_CI_div_G) {
    if (!has_hw) HW = 1;
    if (!has_w_ci) W_CI_div_G = CI_div_G;
    if (!has_w_co) W_CO_div_G = CO_div_G;

    #pragma unroll
    for (int j = 0; j < BLOCK_B_SIZE; j += BLOCK_SIZE){
        int b_hw = blockIdx.x * BLOCK_B_SIZE + j + (has_hw ? threadIdx.x : threadIdx.y);
        int b = has_hw ? b_hw / HW : b_hw;
        int sub_batch = b / SUB_BATCH;
        int hw = has_hw ? b_hw % HW : 0;
        int co = blockIdx.y * BLOCK_CO_SIZE + (has_hw ? threadIdx.y : threadIdx.x);
        int offset = ((b * G + blockIdx.z) * CO_div_G + co) * HW + hw;
        #pragma unroll
        for (int i = 0; i < BLOCK_CO_SIZE; i += BLOCK_SIZE){
            int out_channel = co + i;
            if (b < B && out_channel < CO_div_G) {
                int pos_reg = pos[offset + i * HW];
                float grad = grad_output[offset + i * HW];
                int index = pos_reg & (~(1 << 31));
                float value = pos_reg >= 0 ? grad : -grad;
                atomicAdd(&grad_input[((b * G + blockIdx.z) * CI_div_G + index) * HW + hw], value);
                if (has_w_co) out_channel = w_index_co[(sub_batch * G + blockIdx.z) * CO_div_G + out_channel];
                if (has_w_ci) index = w_index_ci[(sub_batch * G + blockIdx.z) * CI_div_G + index];
                atomicAdd(&grad_weight[(blockIdx.z * W_CO_div_G + out_channel) * W_CI_div_G + index], -value);
            }
        }
    }
}

create_helper(forward, inf_dist)
create_helper(backward_input, inf_dist)
create_helper(backward_input_weight, inf_dist)

void inf_dist::forward_nograd(const float* input, const float* weight,
                              int B, int CO, int CI, int G, int HW, float* output) {
    const int GROUP_CI = 1;
    const int GROUP_CO = 4;
    const int GROUP_B = 4;
    int CI_div_G = CI / G, CO_div_G = CO / G;
    bool has_hw = HW > 1;
    bool check_ci = CI_div_G % (GROUP_CI * BLOCK_SIZE) != 0;

    int num_block_co = (CO_div_G - 1) / (BLOCK_SIZE * GROUP_CO) + 1;
    int num_block_b = (B * HW - 1) / (BLOCK_SIZE * GROUP_B) + 1;
    int ci_split = 1;
    if (!has_hw) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        int ci_split1 = max(deviceProp.multiProcessorCount * 36 / (num_block_co * num_block_b * G), 1);
        ci_split1 = ci_split1 <= 2 ? ci_split1 : ci_split1 <= 6 ? 4 : 8;
        ci_split = min(ci_split1, (CI_div_G - 1) / BLOCK_SIZE + 1);
    }

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(num_block_b, num_block_co, G * ci_split);
    auto tuple = std::make_tuple(has_hw, check_ci, ci_split > 1);
	if (ci_split > 1) cudaMemset(output, 0, B * CO * HW * sizeof(float));
    Call<decltype(tuple)>::call<forward_helper>(std::integer_sequence<int, GROUP_CI, GROUP_CO, GROUP_B>{}, tuple,
        dimGrid, dimBlock,
        input, weight, B, CO_div_G, CI_div_G, HW, G, output);
}

void inf_dist::forward(const float* input, const float* weight,
                       int B, int CO, int CI, int G, int HW, float* output, int* pos,
                       const int* w_index_ci, const int* w_index_co, int W_CO, int W_CI) {
    const int GROUP_CI = 1;
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
    Call<decltype(tuple)>::call<forward_helper>(std::integer_sequence<int, GROUP_CI, GROUP_CO, GROUP_B>{}, tuple,
        dimGrid, dimBlock,
        input, weight, B, CO_div_G, CI_div_G, HW, G, output, pos, w_index_ci, w_index_co, W_CO / G, W_CI / G);
}

void inf_dist::backward_input(const float* grad_output, const int* pos,
                              int B, int CO, int CI, int G, int HW, float* grad_input) {
    int CI_div_G = CI / G, CO_div_G = CO / G;
    bool has_hw = HW > 1;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B * HW - 1) / MAX_BLOCK_B_SIZE + 1, (CO_div_G - 1) / MAX_BLOCK_CO_SIZE + 1, G);
    auto tuple = std::make_tuple(has_hw);
    cudaMemset(grad_input, 0, B * CI * HW * sizeof(float));
    Call<decltype(tuple)>::call<backward_input_helper>(std::integer_sequence<int, MAX_BLOCK_CO_SIZE, MAX_BLOCK_B_SIZE>{},
        tuple, dimGrid, dimBlock,
        grad_output, pos, B, CO_div_G, CI_div_G, HW, G, grad_input);
}

void inf_dist::backward_input_weight(const float* grad_output, const int* pos,
                                     int B, int CO, int CI, int G, int HW, float* grad_input, float* grad_weight,
                                     const int* w_index_ci, const int* w_index_co, int W_CO, int W_CI) {
    int CI_div_G = CI / G, CO_div_G = CO / G;
    bool has_hw = HW > 1;
    bool has_w_ci = w_index_ci != nullptr;
    bool has_w_co = w_index_co != nullptr;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B * HW - 1) / MAX_BLOCK_B_SIZE + 1, (CO_div_G - 1) / MAX_BLOCK_CO_SIZE + 1, G);
    auto tuple = std::make_tuple(has_hw, has_w_ci, has_w_co);
    cudaMemset(grad_input, 0, B * CI * HW * sizeof(float));
    cudaMemset(grad_weight, 0, W_CO * (W_CI / G) * sizeof(float));
    Call<decltype(tuple)>::call<backward_input_weight_helper>(std::integer_sequence<int, MAX_BLOCK_CO_SIZE, MAX_BLOCK_B_SIZE>{},
        tuple, dimGrid, dimBlock,
        grad_output, pos, B, CO_div_G, CI_div_G, HW, G, grad_input, grad_weight, w_index_ci, w_index_co,
        W_CO / G, W_CI / G);
}
