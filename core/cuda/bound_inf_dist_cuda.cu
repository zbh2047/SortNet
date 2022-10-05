#include "norm_dist.h"

const int BLOCK_SIZE = 16;
const int WARP_SIZE = 32;
const int MAX_BLOCK_CO_SIZE = 32;
const int MAX_BLOCK_B_SIZE = 16;

#define CONST_PTR const float* __restrict__
#define PTR float* __restrict__

#define EPS 1e-10f
#define NAN_ID 0xffffffff

// To maximize performance, adjust GROUP_CO, GROUP_B, GROUP_CI and #pragma unroll count for variable t
template <bool has_hw, bool check_ci, int GROUP_CI, int GROUP_CO, int GROUP_B> __global__
void bound_inf_dist_forward_kernel(CONST_PTR inputL, CONST_PTR inputU, CONST_PTR weight,
                                   int B, int CO_div_G, int CI_div_G, int HW, int G,
                                   PTR outputL, PTR outputU) {
    int b_hw = blockIdx.x * (BLOCK_SIZE * GROUP_B) + (has_hw ? threadIdx.x : threadIdx.y);
    int b[GROUP_B], hw[GROUP_B];
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        int b_hw_i = b_hw + i * BLOCK_SIZE;
        if (has_hw) { b[i] = b_hw_i / HW; hw[i] = b_hw_i % HW; }
        else { b[i] = b_hw_i; hw[i] = 0; }
    }

    int write_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + (has_hw ? threadIdx.y : threadIdx.x);
    int read_w_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + threadIdx.y;

    __shared__ float blockIL[GROUP_CI][GROUP_B][BLOCK_SIZE][BLOCK_SIZE]; // CI * B if has_hw else B * CI
    __shared__ float blockIU[GROUP_CI][GROUP_B][BLOCK_SIZE][BLOCK_SIZE]; // CI * B if has_hw else B * CI
    __shared__ float blockW[GROUP_CI][BLOCK_SIZE][BLOCK_SIZE * GROUP_CO + WARP_SIZE / BLOCK_SIZE]; // CI * CO

    float min_outputL[GROUP_B][GROUP_CO], max_outputU[GROUP_B][GROUP_CO];

    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        #pragma unroll
        for (int j = 0; j < GROUP_CO; j++) {
            min_outputL[i][j] = -EPS;
            max_outputU[i][j] = EPS;
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
                    float xL_read, xU_read;
                    if (check_ci) {
                        xL_read = channel < CI_div_G ? inputL[input_offset] : 0;
                        xU_read = channel < CI_div_G ? inputU[input_offset] : 0;
                    }
                    else {
                        xL_read = inputL[input_offset];
                        xU_read = inputU[input_offset];
                    }
                    if (xL_read > xU_read) xL_read = xU_read = (xL_read + xU_read) * 0.5f;
                    blockIL[kk][i][threadIdx.y][threadIdx.x] = xL_read;
                    blockIU[kk][i][threadIdx.y][threadIdx.x] = xU_read;
                }
            }
        }
        #pragma unroll
        for (int kk = 0; kk < GROUP_CI; kk++) {
            #pragma unroll
            for (int i = 0; i < GROUP_CO; i++) {
                if (read_w_co + i * BLOCK_SIZE < CO_div_G) {
                    int channel = k + kk * BLOCK_SIZE + threadIdx.x;
                    int weight_offset = (blockIdx.z * CO_div_G + read_w_co + i * BLOCK_SIZE) * CI_div_G + channel;
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
                        float xL = has_hw ? blockIL[kk][i][t][threadIdx.x] : blockIL[kk][i][threadIdx.y][t];
                        float xU = has_hw ? blockIU[kk][i][t][threadIdx.x] : blockIU[kk][i][threadIdx.y][t];
                        float w = blockW[kk][t][(has_hw ? threadIdx.y : threadIdx.x) + j * BLOCK_SIZE];
                        float t1 = w - xL, t2 = xU - w;
                        min_outputL[i][j] = min(min_outputL[i][j], min(t1, t2));
                        max_outputU[i][j] = max(max_outputU[i][j], max(t1, t2));
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
                    float ansL = -min_outputL[i][j];
                    float ansU = max_outputU[i][j];
                    outputL[((b[i] * G + blockIdx.z) * CO_div_G + channel) * HW + hw[i]] = ansL;
                    outputU[((b[i] * G + blockIdx.z) * CO_div_G + channel) * HW + hw[i]] = ansU;
                }
            }
        }
    }
}

// To maximize performance, adjust GROUP_CO, GROUP_B, GROUP_CI and #pragma unroll count for variable t
template <bool has_hw, bool check_ci, int GROUP_CI, int GROUP_CO, int GROUP_B> __global__
void bound_inf_dist_forward_kernel(CONST_PTR inputL, CONST_PTR inputU, CONST_PTR weight,
                                   int B, int CO_div_G, int CI_div_G, int HW, int G,
                                   PTR outputL, PTR outputU, int* __restrict__ posL, int* __restrict__ posU) {
    int b_hw = blockIdx.x * (BLOCK_SIZE * GROUP_B) + (has_hw ? threadIdx.x : threadIdx.y);
    int b[GROUP_B], hw[GROUP_B];
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        int b_hw_i = b_hw + i * BLOCK_SIZE;
        if (has_hw) { b[i] = b_hw_i / HW; hw[i] = b_hw_i % HW; }
        else { b[i] = b_hw_i; hw[i] = 0; }
    }

    int write_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + (has_hw ? threadIdx.y : threadIdx.x);
    int read_w_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + threadIdx.y;

    __shared__ float blockIL[GROUP_CI][GROUP_B][BLOCK_SIZE][BLOCK_SIZE]; // CI * B if has_hw else B * CI
    __shared__ float blockIU[GROUP_CI][GROUP_B][BLOCK_SIZE][BLOCK_SIZE]; // CI * B if has_hw else B * CI
    __shared__ float blockW[GROUP_CI][BLOCK_SIZE][BLOCK_SIZE * GROUP_CO + WARP_SIZE / BLOCK_SIZE]; // CI * CO

    float min_outputL[GROUP_B][GROUP_CO], max_outputU[GROUP_B][GROUP_CO];
    int res_posL[GROUP_B][GROUP_CO], res_posU[GROUP_B][GROUP_CO];

    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        #pragma unroll
        for (int j = 0; j < GROUP_CO; j++) {
            min_outputL[i][j] = -EPS;
            max_outputU[i][j] = EPS;
            res_posL[i][j] = NAN_ID;
            res_posU[i][j] = 0;
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
                    float xL_read, xU_read;
                    if (check_ci) {
                        xL_read = channel < CI_div_G ? inputL[input_offset] : 0;
                        xU_read = channel < CI_div_G ? inputU[input_offset] : 0;
                    }
                    else {
                        xL_read = inputL[input_offset];
                        xU_read = inputU[input_offset];
                    }
                    if (xL_read > xU_read) xL_read = xU_read = (xL_read + xU_read) * 0.5f;
                    blockIL[kk][i][threadIdx.y][threadIdx.x] = xL_read;
                    blockIU[kk][i][threadIdx.y][threadIdx.x] = xU_read;
                }
            }
        }
        #pragma unroll
        for (int kk = 0; kk < GROUP_CI; kk++) {
            #pragma unroll
            for (int i = 0; i < GROUP_CO; i++) {
                if (read_w_co + i * BLOCK_SIZE < CO_div_G) {
                    int channel = k + kk * BLOCK_SIZE + threadIdx.x;
                    int weight_offset = (blockIdx.z * CO_div_G + read_w_co + i * BLOCK_SIZE) * CI_div_G + channel;
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
                        float xL = has_hw ? blockIL[kk][i][t][threadIdx.x] : blockIL[kk][i][threadIdx.y][t];
                        float xU = has_hw ? blockIU[kk][i][t][threadIdx.x] : blockIU[kk][i][threadIdx.y][t];
                        float w = blockW[kk][t][(has_hw ? threadIdx.y : threadIdx.x) + j * BLOCK_SIZE];
                        float t1 = w - xL, t2 = xU - w;
                        float tmp1 = min(t1, t2), tmp2 = max(t1, t2);
                        int pos = k + kk * BLOCK_SIZE + t + (t1 <= t2 ? 0 : 1 << 31);
                        if (tmp1 < min_outputL[i][j]) {
                            min_outputL[i][j] = tmp1;
                            res_posL[i][j] = pos;
                        }
                        if (tmp2 > max_outputU[i][j]) {
                            max_outputU[i][j] = tmp2;
                            res_posU[i][j] = pos;
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
                    float ansL = -min_outputL[i][j];
                    float ansU = max_outputU[i][j];
                    int offset = ((b[i] * G + blockIdx.z) * CO_div_G + channel) * HW + hw[i];
                    outputL[offset] = ansL;
                    outputU[offset] = ansU;
                    posL[offset] = res_posL[i][j];
                    posU[offset] = res_posU[i][j];
                }
            }
        }
    }
}

template <bool has_hw, int BLOCK_CO_SIZE, int BLOCK_B_SIZE>
__global__ void bound_inf_dist_backward_input_kernel(CONST_PTR grad_outputL, CONST_PTR grad_outputU,
                                                     const int* __restrict__ posL, const int* __restrict__ posU,
                                                     int B, int CO_div_G, int CI_div_G, int HW, int G,
                                                     PTR grad_inputL, PTR grad_inputU) {
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
                int pos_regL = posL[offset + i * HW], pos_regU = posU[offset + i * HW];
                float gradL = grad_outputL[offset + i * HW], gradU = grad_outputU[offset + i * HW];
                int indexL = pos_regL & (~(1 << 31)), indexU = pos_regU & (~(1 << 31));
                float valueL = pos_regL >= 0 ? gradL : -gradL, valueU = pos_regU >= 0 ? gradU : -gradU;
                PTR ptrL = pos_regL >= 0 ? grad_inputL : grad_inputU;
                PTR ptrU = pos_regU >= 0 ? grad_inputU : grad_inputL;
                if (pos_regL != NAN_ID)
                    atomicAdd(&ptrL[((b * G + blockIdx.z) * CI_div_G + indexL) * HW + hw], valueL);
                atomicAdd(&ptrU[((b * G + blockIdx.z) * CI_div_G + indexU) * HW + hw], valueU);
            }
        }
    }
}

template <bool has_hw, int BLOCK_CO_SIZE, int BLOCK_B_SIZE>
__global__ void bound_inf_dist_backward_input_weight_kernel(CONST_PTR grad_outputL, CONST_PTR grad_outputU,
                                                            const int* __restrict__ posL, const int* __restrict__ posU,
                                                            int B, int CO_div_G, int CI_div_G, int HW, int G,
                                                            PTR grad_inputL, PTR grad_inputU, PTR grad_weight) {
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
                int pos_regL = posL[offset + i * HW], pos_regU = posU[offset + i * HW];
                float gradL = grad_outputL[offset + i * HW], gradU = grad_outputU[offset + i * HW];
                int indexL = pos_regL & (~(1 << 31)), indexU = pos_regU & (~(1 << 31));
                float valueL = pos_regL >= 0 ? gradL : -gradL, valueU = pos_regU >= 0 ? gradU : -gradU;
                PTR ptrL = pos_regL >= 0 ? grad_inputL : grad_inputU;
                PTR ptrU = pos_regU >= 0 ? grad_inputU : grad_inputL;
                if (pos_regL != NAN_ID) {
                    atomicAdd(&ptrL[((b * G + blockIdx.z) * CI_div_G + indexL) * HW + hw], valueL);
                    atomicAdd(&grad_weight[(blockIdx.z * CO_div_G + co + i) * CI_div_G + indexL], -valueL);
                }
                atomicAdd(&ptrU[((b * G + blockIdx.z) * CI_div_G + indexU) * HW + hw], valueU);
                atomicAdd(&grad_weight[(blockIdx.z * CO_div_G + co + i) * CI_div_G + indexU], -valueU);
            }
        }
    }
}

#define GROUP_CO 2
#define GROUP_B 2

#define inf_dist_forward_helper_func(func, GROUP_CO, GROUP_B, has_hw, paras...) \
    int num_block_co = (CO / G - 1) / (BLOCK_SIZE * GROUP_CO) + 1; \
    int num_block_b = (B * HW - 1) / (BLOCK_SIZE * GROUP_B) + 1; \
    int CI_div_G = CI / G; \
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); \
    dim3 dimGrid(num_block_b, num_block_co, G); \
    if (CI_div_G % (1 * BLOCK_SIZE) == 0) \
        func<has_hw, false, 1, GROUP_CO, GROUP_B><<<dimGrid, dimBlock>>>(paras); \
    else func<has_hw, true, 1, GROUP_CO, GROUP_B><<<dimGrid, dimBlock>>>(paras);

void bound_inf_dist::forward_nograd(const float* inputL, const float* inputU, const float* weight,
                                    int B, int CO, int CI, int G, int HW, float* outputL, float* outputU) {
    if (HW == 1) {
        inf_dist_forward_helper_func(bound_inf_dist_forward_kernel, 4, 4, false,
                                     inputL, inputU, weight, B, CO / G, CI_div_G, HW, G, outputL, outputU);
    }
    else if (CO / G <= 2 * BLOCK_SIZE) {
        inf_dist_forward_helper_func(bound_inf_dist_forward_kernel, 2, 4, true,
                                     inputL, inputU, weight, B, CO / G, CI_div_G, HW, G, outputL, outputU);
    }
    else {
        inf_dist_forward_helper_func(bound_inf_dist_forward_kernel, 4, 4, true,
                                     inputL, inputU, weight, B, CO / G, CI_div_G, HW, G, outputL, outputU);
    }
}

void bound_inf_dist::forward(const float* inputL, const float* inputU, const float* weight,
                             int B, int CO, int CI, int G, int HW, float* outputL, float* outputU,
                             int* posL, int* posU) {
    if (HW == 1) {
        inf_dist_forward_helper_func(bound_inf_dist_forward_kernel, 4, 2, false,
                                     inputL, inputU, weight, B, CO / G, CI_div_G, HW, G, outputL, outputU, posL, posU);
    }
    else if (CO / G <= 2 * BLOCK_SIZE) {
        inf_dist_forward_helper_func(bound_inf_dist_forward_kernel, 2, 2, true,
                                     inputL, inputU, weight, B, CO / G, CI_div_G, HW, G, outputL, outputU, posL, posU);
    }
    else {
        inf_dist_forward_helper_func(bound_inf_dist_forward_kernel, 4, 2, true,
                                     inputL, inputU, weight, B, CO / G, CI_div_G, HW, G, outputL, outputU, posL, posU);
    }
}

#undef GROUP_CO
#undef GROUP_B

#define inf_dist_backward_helper_func(func, HW, CO_div_G, paras...) \
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); \
    int num_block_b = (B * HW - 1) / MAX_BLOCK_B_SIZE + 1; \
    if (HW == 1) { \
        dim3 dimGrid2(num_block_b, (CO_div_G - 1) / BLOCK_SIZE + 1, G); \
        func<false, MAX_BLOCK_CO_SIZE, MAX_BLOCK_B_SIZE><<<dimGrid2, dimBlock>>>(paras); \
    } \
    else { \
        dim3 dimGrid2(num_block_b, (CO_div_G - 1) / (2 * BLOCK_SIZE) + 1, G); \
        func<true, MAX_BLOCK_CO_SIZE, MAX_BLOCK_B_SIZE><<<dimGrid2, dimBlock>>>(paras); \
    }

void bound_inf_dist::backward_input(const float* grad_outputL, const float* grad_outputU,
                                    const int* posL, const int* posU, int B, int CO, int CI, int G, int HW,
                                    float* grad_inputL, float* grad_inputU) {
    int CO_div_G = CO / G;
    cudaMemset(grad_inputL, 0, B * CI * HW * sizeof(float));
    cudaMemset(grad_inputU, 0, B * CI * HW * sizeof(float));
    inf_dist_backward_helper_func(bound_inf_dist_backward_input_kernel, HW, CO_div_G,
                                  grad_outputL, grad_outputU, posL, posU, B, CO_div_G, CI / G, HW, G,
                                  grad_inputL, grad_inputU);
}

void bound_inf_dist::backward_input_weight(const float* grad_outputL, const float* grad_outputU,
                                           const int* posL, const int* posU, int B, int CO, int CI, int G, int HW,
                                           float* grad_inputL, float* grad_inputU, float* grad_weight) {
    int CO_div_G = CO / G;
    cudaMemset(grad_inputL, 0, B * CI * HW * sizeof(float));
    cudaMemset(grad_inputU, 0, B * CI * HW * sizeof(float));
    cudaMemset(grad_weight, 0, CO * (CI / G) * sizeof(float));
    inf_dist_backward_helper_func(bound_inf_dist_backward_input_weight_kernel, HW, CO_div_G,
                                  grad_outputL, grad_outputU, posL, posU, B, CO_div_G, CI / G, HW, G,
                                  grad_inputL, grad_inputU, grad_weight);
}
