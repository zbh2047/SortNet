#include <tuple>
#include <cassert>
#include "norm_dist.h"
#include "forward.cuh"

const int BLOCK_SIZE = 16;
const int WARP_SIZE = 32;
const int MAX_BLOCK_CO_SIZE = 32;
const int MAX_BLOCK_B_SIZE = 16;

#define EPS 1e-10f

#define CONST_PTR const float* __restrict__
#define PTR float* __restrict__

template <int GROUP_CO, int GROUP_B, int K, bool has_hw> __global__
void sort_forward_kernel(CONST_PTR input, CONST_PTR weight,
                         int B, int CO_div_G, int CI_div_G, int HW, int G, PTR output, float q) {
    int b_hw = blockIdx.x * (BLOCK_SIZE * GROUP_B) + (has_hw ? threadIdx.x : threadIdx.y);
    int b[GROUP_B], hw[GROUP_B];
    int g = blockIdx.z;
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        int b_hw_i = b_hw + i * BLOCK_SIZE;
        if (has_hw) { b[i] = b_hw_i / HW; hw[i] = b_hw_i % HW; }
        else { b[i] = b_hw_i; hw[i] = 0; }
    }
    if (!has_hw) HW = 1;

    int write_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + (has_hw ? threadIdx.y : threadIdx.x);
    int read_w_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + threadIdx.y;

    __shared__ float blockI[GROUP_B][BLOCK_SIZE][BLOCK_SIZE]; // CI * B if has_hw else B * CI
    __shared__ float blockW[BLOCK_SIZE][BLOCK_SIZE * GROUP_CO + WARP_SIZE / BLOCK_SIZE]; // CI * CO

    float res[GROUP_B][GROUP_CO][K];
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        #pragma unroll
        for (int j = 0; j < GROUP_CO; j++) {
            #pragma unroll
            for (int k = 0; k < K; k++)
                res[i][j][k] = 0;
        }
    }

    for (int s = 0; s < CI_div_G; s += BLOCK_SIZE) {
        #pragma unroll
        for (int i = 0; i < GROUP_B; i++) {
            if (b[i] < B) {
                int channel = s + (has_hw ? threadIdx.y : threadIdx.x);
                int input_offset = ((b[i] * G + g) * CI_div_G + channel) * HW + hw[i];
                blockI[i][threadIdx.y][threadIdx.x] = channel < CI_div_G ? input[input_offset] : 0;
            }
        }
        #pragma unroll
        for (int i = 0; i < GROUP_CO; i++) {
            int out_channel = read_w_co + i * BLOCK_SIZE;
            if (out_channel < CO_div_G) {
                int weight_offset = (g * CO_div_G + out_channel) * CI_div_G + s + threadIdx.x;
                blockW[threadIdx.x][threadIdx.y + i * BLOCK_SIZE] = s + threadIdx.x < CI_div_G ? weight[weight_offset] : 0;
            }
        }
        __syncthreads();
        #pragma unroll(1)
        for (int t = 0; t < BLOCK_SIZE; t++) {
            #pragma unroll
            for (int i = 0; i < GROUP_B; i++) {
                #pragma unroll
                for (int j = 0; j < GROUP_CO; j++) {
                    float x = has_hw ? blockI[i][t][threadIdx.x] : blockI[i][threadIdx.y][t];
                    float w = blockW[t][(has_hw ? threadIdx.y : threadIdx.x) + j * BLOCK_SIZE];
                    float value = abs(x - w);
                    #pragma unroll
                    for (int k = 0; k < K - 1; k++)
                        res[i][j][k] = max(res[i][j][k], min(res[i][j][k + 1], value));
                    res[i][j][K - 1] = max(res[i][j][K - 1], value);
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
                    float ans = EPS, c = 1;
                    #pragma unroll
                    for (int k = K - 1; k >= 0; k--) {
                        ans += res[i][j][k] * c * q;
                        c *= 1 - q;
                    }
                    output[offset] = ans / (1 - c);
                }
            }
        }
    }
}

__device__ __forceinline__ void update(float& res_x, int& res_p, float x, int p) {
    res_x = x; res_p = p;
}

__device__ __forceinline__ bool update(float& res_x, int& res_p, float x1, int p1, float x2, int p2) {
    float x = x1 < x2 ? x1 : x2;
    int p = x1 < x2 ? p1 : p2;
    update (res_x, res_p, x, p);
    return x1 < x2;
}

template <int GROUP_CO, int GROUP_B, int K, bool has_hw> __global__
void sort_forward_kernel(CONST_PTR input, CONST_PTR weight,
                         int B, int CO_div_G, int CI_div_G, int HW, int G, PTR output, int* __restrict__ pos, float q) {
    int b_hw = blockIdx.x * (BLOCK_SIZE * GROUP_B) + (has_hw ? threadIdx.x : threadIdx.y);
    int b[GROUP_B], hw[GROUP_B];
    int g = blockIdx.z;
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        int b_hw_i = b_hw + i * BLOCK_SIZE;
        if (has_hw) { b[i] = b_hw_i / HW; hw[i] = b_hw_i % HW; }
        else { b[i] = b_hw_i; hw[i] = 0; }
    }
    if (!has_hw) HW = 1;

    int write_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + (has_hw ? threadIdx.y : threadIdx.x);
    int read_w_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + threadIdx.y;

    __shared__ float blockI[GROUP_B][BLOCK_SIZE][BLOCK_SIZE]; // CI * B if has_hw else B * CI
    __shared__ float blockW[BLOCK_SIZE][BLOCK_SIZE * GROUP_CO + WARP_SIZE / BLOCK_SIZE]; // CI * CO

    float res[GROUP_B][GROUP_CO][K];
    int index[GROUP_B][GROUP_CO][K];
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        #pragma unroll
        for (int j = 0; j < GROUP_CO; j++) {
            #pragma unroll
            for (int k = 0; k < K; k++) {
                res[i][j][k] = 0;
                index[i][j][k] = 0;
            }
        }
    }

    for (int s = 0; s < CI_div_G; s += BLOCK_SIZE) {
        #pragma unroll
        for (int i = 0; i < GROUP_B; i++) {
            if (b[i] < B) {
                int channel = s + (has_hw ? threadIdx.y : threadIdx.x);
                int input_offset = ((b[i] * G + g) * CI_div_G + channel) * HW + hw[i];
                blockI[i][threadIdx.y][threadIdx.x] = channel < CI_div_G ? input[input_offset] : 0;
            }
        }
        #pragma unroll
        for (int i = 0; i < GROUP_CO; i++) {
            int out_channel = read_w_co + i * BLOCK_SIZE;
            if (out_channel < CO_div_G) {
                int weight_offset = (g * CO_div_G + out_channel) * CI_div_G + s + threadIdx.x;
                blockW[threadIdx.x][threadIdx.y + i * BLOCK_SIZE] = s + threadIdx.x < CI_div_G ? weight[weight_offset] : 0;
            }
        }
        __syncthreads();
        #pragma unroll(1)
        for (int t = 0; t < BLOCK_SIZE; t++) {
            #pragma unroll
            for (int i = 0; i < GROUP_B; i++) {
                #pragma unroll
                for (int j = 0; j < GROUP_CO; j++) {
                    float x = has_hw ? blockI[i][t][threadIdx.x] : blockI[i][threadIdx.y][t];
                    float w = blockW[t][(has_hw ? threadIdx.y : threadIdx.x) + j * BLOCK_SIZE];
                    float tmp = x - w, value = abs(tmp);
                    if (value > res[i][j][0]) {
                        int cur_pos = s + t + (tmp >= 0 ? 0 : 1 << 31);
                        int k;
                        #pragma unroll
                        for (k = 0; k < K - 1; k++) {
                            if (update(res[i][j][k], index[i][j][k], value, cur_pos, res[i][j][k + 1], index[i][j][k + 1]))
                                break;
                        }
                        if (k == K - 1) update(res[i][j][k], index[i][j][k], value, cur_pos);
                    }
                }
            }
        }
        __syncthreads();
    }
    int size = B * G * CO_div_G * HW;
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        if (b[i] < B) {
            #pragma unroll
            for (int j = 0; j < GROUP_CO; j++) {
                int channel = write_co + j * BLOCK_SIZE;
                if (channel < CO_div_G) {
                    int offset = ((b[i] * G + g) * CO_div_G + channel) * HW + hw[i];
                    float ans = EPS, c = 1;
                    #pragma unroll
                    for (int k = K - 1; k >= 0; k--) {
                        ans += res[i][j][k] * c * q;
                        c *= 1 - q;
                    }
                    output[offset] = ans / (1 - c);
                    #pragma unroll
                    for (int k = K - 1; k >= 0; k--)
                        pos[(K - 1 - k) * size + offset] = index[i][j][k];
                }
            }
        }
    }
}

template <int GROUP_CO, int GROUP_B, int K, bool has_hw> __global__
void sort_bound_forward_kernel(CONST_PTR inputL, CONST_PTR inputU, CONST_PTR weight,
                               int B, int CO_div_G, int CI_div_G, int HW, int G, PTR outputL, PTR outputU, float q) {
    int b_hw = blockIdx.x * (BLOCK_SIZE * GROUP_B) + (has_hw ? threadIdx.x : threadIdx.y);
    int b[GROUP_B], hw[GROUP_B];
    int g = blockIdx.z;
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        int b_hw_i = b_hw + i * BLOCK_SIZE;
        if (has_hw) { b[i] = b_hw_i / HW; hw[i] = b_hw_i % HW; }
        else { b[i] = b_hw_i; hw[i] = 0; }
    }
    if (!has_hw) HW = 1;

    int write_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + (has_hw ? threadIdx.y : threadIdx.x);
    int read_w_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + threadIdx.y;

    __shared__ float blockIL[GROUP_B][BLOCK_SIZE][BLOCK_SIZE]; // CI * B if has_hw else B * CI
    __shared__ float blockIU[GROUP_B][BLOCK_SIZE][BLOCK_SIZE]; // CI * B if has_hw else B * CI
    __shared__ float blockW[BLOCK_SIZE][BLOCK_SIZE * GROUP_CO + WARP_SIZE / BLOCK_SIZE]; // CI * CO

    float resL[GROUP_B][GROUP_CO][K], resU[GROUP_B][GROUP_CO][K];
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        #pragma unroll
        for (int j = 0; j < GROUP_CO; j++) {
            #pragma unroll
            for (int k = 0; k < K; k++)
                resL[i][j][k] = resU[i][j][k] = 0;
        }
    }

    for (int s = 0; s < CI_div_G; s += BLOCK_SIZE) {
        #pragma unroll
        for (int i = 0; i < GROUP_B; i++) {
            if (b[i] < B) {
                int channel = s + (has_hw ? threadIdx.y : threadIdx.x);
                int input_offset = ((b[i] * G + g) * CI_div_G + channel) * HW + hw[i];
                float xL_read = channel < CI_div_G ? inputL[input_offset] : 0;
                float xU_read = channel < CI_div_G ? inputU[input_offset] : 0;
                if (xL_read > xU_read) xL_read = xU_read = (xL_read + xU_read) * 0.5f;
                blockIL[i][threadIdx.y][threadIdx.x] = xL_read;
                blockIU[i][threadIdx.y][threadIdx.x] = xU_read;
            }
        }
        #pragma unroll
        for (int i = 0; i < GROUP_CO; i++) {
            int out_channel = read_w_co + i * BLOCK_SIZE;
            if (out_channel < CO_div_G) {
                int weight_offset = (g * CO_div_G + out_channel) * CI_div_G + s + threadIdx.x;
                blockW[threadIdx.x][threadIdx.y + i * BLOCK_SIZE] = s + threadIdx.x < CI_div_G ? weight[weight_offset] : 0;
            }
        }
        __syncthreads();
        #pragma unroll(1)
        for (int t = 0; t < BLOCK_SIZE; t++) {
            #pragma unroll
            for (int i = 0; i < GROUP_B; i++) {
                #pragma unroll
                for (int j = 0; j < GROUP_CO; j++) {
                    float xL = has_hw ? blockIL[i][t][threadIdx.x] : blockIL[i][threadIdx.y][t];
                    float xU = has_hw ? blockIU[i][t][threadIdx.x] : blockIU[i][threadIdx.y][t];
                    float w = blockW[t][(has_hw ? threadIdx.y : threadIdx.x) + j * BLOCK_SIZE];
                    float t1 = w - xL, t2 = xU - w;
                    float valueL = min(t1, t2), valueU = max(t1, t2);
                    #pragma unroll
                    for (int k = 0; k < K - 1; k++) {
                        resL[i][j][k] = min(resL[i][j][k], max(resL[i][j][k + 1], valueL));
                        resU[i][j][k] = max(resU[i][j][k], min(resU[i][j][k + 1], valueU));
                    }
                    resL[i][j][K - 1] = min(resL[i][j][K - 1], valueL);
                    resU[i][j][K - 1] = max(resU[i][j][K - 1], valueU);
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
                    float ansL = -EPS, ansU = EPS, c = 1;
                    #pragma unroll
                    for (int k = K - 1; k >= 0; k--) {
                        ansL += resL[i][j][k] * c * q;
                        ansU += resU[i][j][k] * c * q;
                        c *= 1 - q;
                    }
                    outputL[offset] = -ansL / (1 - c);
                    outputU[offset] = ansU / (1 - c);
                }
            }
        }
    }
}

template <int BLOCK_CO_SIZE, int BLOCK_B_SIZE, bool has_hw>
__global__ void sort_backward_input_kernel(CONST_PTR grad_output, const int* __restrict__ pos,
                                            int B, int CO_div_G, int CI_div_G, int HW, int G, PTR grad_input,
                                            float q, int K) {
    if (!has_hw) HW = 1;
    int size = B * G * CO_div_G * HW;
    float normalize = powf(1 - q, K); normalize = 1.0f + normalize / (1.0f - normalize);
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
                float grad = grad_output[offset + i * HW], c = normalize;
                for (int k = 0; k < K; k++) {
                    int pos_reg = pos[size * k + offset + i * HW];
                    int index = pos_reg & (~(1 << 31));
                    float value = (pos_reg >= 0 ? grad : -grad) * c * q;
                    atomicAdd(&grad_input[((b * G + blockIdx.z) * CI_div_G + index) * HW + hw], value);
                    c *= 1 - q;
                }
            }
        }
    }
}

template <int BLOCK_CO_SIZE, int BLOCK_B_SIZE, bool has_hw>
__global__ void sort_backward_input_weight_kernel(CONST_PTR grad_output, const int* __restrict__ pos,
                                                  int B, int CO_div_G, int CI_div_G, int HW, int G,
                                                  PTR grad_input, PTR grad_weight, float q, int K) {
    if (!has_hw) HW = 1;
    int size = B * G * CO_div_G * HW;
    float normalize = powf(1 - q, K); normalize = 1.0f + normalize / (1.0f - normalize);
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
                float grad = grad_output[offset + i * HW], c = normalize;
                for (int k = 0; k < K; k++) {
                    int pos_reg = pos[size * k + offset + i * HW];
                    int index = pos_reg & (~(1 << 31));
                    float value = (pos_reg >= 0 ? grad : -grad) * c * q;
                    atomicAdd(&grad_input[((b * G + blockIdx.z) * CI_div_G + index) * HW + hw], value);
                    atomicAdd(&grad_weight[(blockIdx.z * CO_div_G + co + i) * CI_div_G + index], -value);
                    c *= 1 - q;
                }
            }
        }
    }
}

create_helper(forward, sort)
create_helper(bound_forward, sort)
create_helper(backward_input, sort)
create_helper(backward_input_weight, sort)

template<typename F, typename Tuple, size_t ... I>
auto call(F f, Tuple t, std::index_sequence<I...>) {
     return f(std::get<I>(t)...);
}

template <int GROUP_CO, int GROUP_B, int K>
void helper_nograd(const float* input, const float* weight,
            int B, int CO, int CI, int G, int HW, float* output, float q) {

    int CI_div_G = CI / G, CO_div_G = CO / G;
    bool has_hw = HW > 1;
    int num_block_co = (CO_div_G - 1) / (BLOCK_SIZE * GROUP_CO) + 1;
    int num_block_b = (B * HW - 1) / (BLOCK_SIZE * GROUP_B) + 1;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(num_block_b, num_block_co, G);
    auto tuple = std::make_tuple(has_hw);
    Call<decltype(tuple)>::call<forward_helper>(std::integer_sequence<int, GROUP_CO, GROUP_B, K>{}, tuple,
        dimGrid, dimBlock,
        input, weight, B, CO_div_G, CI_div_G, HW, G, output, q);
}

void sort_neuron::forward_nograd(const float* input, const float* weight,
                                 int B, int CO, int CI, int G, int HW, float* output, float q, int K) {
    auto paras = std::make_tuple(input, weight, B, CO, CI, G, HW, output, q);
    auto _ = std::make_index_sequence<std::tuple_size<decltype(paras)>::value>{};
    switch (K) {
    case 1: assert(((void)"Please call L-infinity distance function directly", false)); return;
    case 2: return call(helper_nograd<4, 2, 2>, paras, _);
    case 3: return call(helper_nograd<4, 2, 3>, paras, _);
    case 4: return call(helper_nograd<4, 2, 4>, paras, _);
    case 5: return call(helper_nograd<4, 2, 5>, paras, _);
    case 6: return call(helper_nograd<3, 2, 6>, paras, _);
    case 7: return call(helper_nograd<3, 2, 7>, paras, _);
    case 8: return call(helper_nograd<3, 2, 8>, paras, _);
    case 9: return call(helper_nograd<3, 2, 9>, paras, _);
    case 10: return call(helper_nograd<3, 2, 10>, paras, _);
    case 11: return call(helper_nograd<2, 2, 11>, paras, _);
    case 12: return call(helper_nograd<2, 2, 12>, paras, _);
    case 13: return call(helper_nograd<2, 2, 13>, paras, _);
    case 14: return call(helper_nograd<2, 2, 14>, paras, _);
    case 15: return call(helper_nograd<2, 2, 15>, paras, _);
    case 16: return call(helper_nograd<2, 2, 16>, paras, _);
    case 17: return call(helper_nograd<1, 2, 17>, paras, _);
    case 18: return call(helper_nograd<1, 2, 18>, paras, _);
    case 19: return call(helper_nograd<1, 2, 19>, paras, _);
    case 20: return call(helper_nograd<1, 2, 20>, paras, _);
    case 21: return call(helper_nograd<1, 2, 21>, paras, _);
    case 22: return call(helper_nograd<1, 2, 22>, paras, _);
    case 23: return call(helper_nograd<1, 2, 23>, paras, _);
    case 24: return call(helper_nograd<1, 2, 24>, paras, _);
    case 25: return call(helper_nograd<1, 2, 25>, paras, _);
    default: assert(((void)"K greater than 25 is not supported.", false));
    }
}

template <int GROUP_CO, int GROUP_B, int K>
void helper(const float* input, const float* weight,
            int B, int CO, int CI, int G, int HW, float* output, int* pos, float q) {

    int CI_div_G = CI / G, CO_div_G = CO / G;
    bool has_hw = HW > 1;
    int num_block_co = (CO_div_G - 1) / (BLOCK_SIZE * GROUP_CO) + 1;
    int num_block_b = (B * HW - 1) / (BLOCK_SIZE * GROUP_B) + 1;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(num_block_b, num_block_co, G);
    auto tuple = std::make_tuple(has_hw);
    Call<decltype(tuple)>::call<forward_helper>(std::integer_sequence<int, GROUP_CO, GROUP_B, K>{}, tuple,
        dimGrid, dimBlock,
        input, weight, B, CO_div_G, CI_div_G, HW, G, output, pos, q);
}

void sort_neuron::forward(const float* input, const float* weight,
                          int B, int CO, int CI, int G, int HW, float* output, int* pos, float q, int K) {
    auto paras = std::make_tuple(input, weight, B, CO, CI, G, HW, output, pos, q);
    auto _ = std::make_index_sequence<std::tuple_size<decltype(paras)>::value>{};
    switch (K) {
    case 1: assert(((void)"Please call L-infinity distance function directly", false)); return;
    case 2: return call(helper<4, 2, 2>, paras, _);
    case 3: return call(helper<3, 2, 3>, paras, _);
    case 4: return call(helper<3, 2, 4>, paras, _);
    case 5: return call(helper<3, 2, 5>, paras, _);
    case 6: return call(helper<2, 2, 6>, paras, _);
    case 7: return call(helper<2, 2, 7>, paras, _);
    case 8: return call(helper<2, 2, 8>, paras, _);
    case 9: return call(helper<3, 1, 9>, paras, _);
    case 10: return call(helper<3, 1, 10>, paras, _);
    case 11: return call(helper<3, 1, 11>, paras, _);
    case 12: return call(helper<2, 1, 12>, paras, _);
    case 13: return call(helper<2, 1, 13>, paras, _);
    case 14: return call(helper<2, 1, 14>, paras, _);
    case 15: return call(helper<2, 1, 15>, paras, _);
    case 16: return call(helper<2, 1, 16>, paras, _);
    case 17: return call(helper<1, 1, 17>, paras, _);
    case 18: return call(helper<1, 1, 18>, paras, _);
    case 19: return call(helper<1, 1, 19>, paras, _);
    case 20: return call(helper<1, 1, 20>, paras, _);
    case 21: return call(helper<1, 1, 21>, paras, _);
    case 22: return call(helper<1, 1, 22>, paras, _);
    case 23: return call(helper<1, 1, 23>, paras, _);
    case 24: return call(helper<1, 1, 24>, paras, _);
    case 25: return call(helper<1, 1, 25>, paras, _);
    default: assert(((void)"K greater than 25 is not supported.", false));
    }
}

template <int GROUP_CO, int GROUP_B, int K>
void bound_helper(const float* inputL, const float* inputU, const float* weight,
            int B, int CO, int CI, int G, int HW, float* outputL, float* outputU, float q) {
    int CI_div_G = CI / G, CO_div_G = CO / G;
    bool has_hw = HW > 1;
    int num_block_co = (CO_div_G - 1) / (BLOCK_SIZE * GROUP_CO) + 1;
    int num_block_b = (B * HW - 1) / (BLOCK_SIZE * GROUP_B) + 1;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(num_block_b, num_block_co, G);
    auto tuple = std::make_tuple(has_hw);
    Call<decltype(tuple)>::call<bound_forward_helper>(std::integer_sequence<int, GROUP_CO, GROUP_B, K>{}, tuple,
        dimGrid, dimBlock,
        inputL, inputU, weight, B, CO_div_G, CI_div_G, HW, G, outputL, outputU, q);
}

void bound_sort_neuron_forward_cuda(const float* inputL, const float* inputU, const float* weight,
                                    int B, int CO, int CI, int G, int HW, float* outputL, float* outputU, float q, int K) {
    auto paras = std::make_tuple(inputL, inputU, weight, B, CO, CI, G, HW, outputL, outputU, q);
    auto _ = std::make_index_sequence<std::tuple_size<decltype(paras)>::value>{};
    switch (K) {
    case 1: assert(((void)"Please call L-infinity distance function directly", false)); return;
    case 2: return call(bound_helper<4, 2, 2>, paras, _);
    case 3: return call(bound_helper<3, 2, 3>, paras, _);
    case 4: return call(bound_helper<3, 2, 4>, paras, _);
    case 5: return call(bound_helper<3, 2, 5>, paras, _);
    case 6: return call(bound_helper<2, 2, 6>, paras, _);
    case 7: return call(bound_helper<2, 2, 7>, paras, _);
    case 8: return call(bound_helper<2, 2, 8>, paras, _);
    case 9: return call(bound_helper<3, 1, 9>, paras, _);
    case 10: return call(bound_helper<3, 1, 10>, paras, _);
    case 11: return call(bound_helper<3, 1, 11>, paras, _);
    case 12: return call(bound_helper<2, 1, 12>, paras, _);
    case 13: return call(bound_helper<2, 1, 13>, paras, _);
    case 14: return call(bound_helper<2, 1, 14>, paras, _);
    case 15: return call(bound_helper<2, 1, 15>, paras, _);
    case 16: return call(bound_helper<2, 1, 16>, paras, _);
    case 17: return call(bound_helper<1, 1, 17>, paras, _);
    case 18: return call(bound_helper<1, 1, 18>, paras, _);
    case 19: return call(bound_helper<1, 1, 19>, paras, _);
    case 20: return call(bound_helper<1, 1, 20>, paras, _);
    case 21: return call(bound_helper<1, 1, 21>, paras, _);
    case 22: return call(bound_helper<1, 1, 22>, paras, _);
    case 23: return call(bound_helper<1, 1, 23>, paras, _);
    case 24: return call(bound_helper<1, 1, 24>, paras, _);
    case 25: return call(bound_helper<1, 1, 25>, paras, _);
    default: assert(((void)"K greater than 25 is not supported.", false));
    }
}

void sort_neuron::backward_input(const float* grad_output, const int* pos,
                                 int B, int CO, int CI, int G, int HW, float* grad_input, float q, int K) {
    int CI_div_G = CI / G, CO_div_G = CO / G;
    bool has_hw = HW > 1;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B * HW - 1) / MAX_BLOCK_B_SIZE + 1, (CO_div_G - 1) / MAX_BLOCK_CO_SIZE + 1, G);
    auto tuple = std::make_tuple(has_hw);
    cudaMemset(grad_input, 0, B * CI * HW * sizeof(float));
    Call<decltype(tuple)>::call<backward_input_helper>(std::integer_sequence<int, MAX_BLOCK_CO_SIZE, MAX_BLOCK_B_SIZE>{},
        tuple, dimGrid, dimBlock,
        grad_output, pos, B, CO_div_G, CI_div_G, HW, G, grad_input, q, K);
}

void sort_neuron::backward_input_weight(const float* grad_output, const int* pos,
                                        int B, int CO, int CI, int G, int HW, float* grad_input, float* grad_weight,
                                        float q, int K) {
    int CI_div_G = CI / G, CO_div_G = CO / G;
    bool has_hw = HW > 1;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B * HW - 1) / MAX_BLOCK_B_SIZE + 1, (CO_div_G - 1) / MAX_BLOCK_CO_SIZE + 1, G);
    auto tuple = std::make_tuple(has_hw);
    cudaMemset(grad_input, 0, B * CI * HW * sizeof(float));
    cudaMemset(grad_weight, 0, CO * (CI / G) * sizeof(float));
    Call<decltype(tuple)>::call<backward_input_weight_helper>(std::integer_sequence<int, MAX_BLOCK_CO_SIZE, MAX_BLOCK_B_SIZE>{},
        tuple, dimGrid, dimBlock,
        grad_output, pos, B, CO_div_G, CI_div_G, HW, G, grad_input, grad_weight, q, K);
}
