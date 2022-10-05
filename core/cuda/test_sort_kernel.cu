#include "norm_dist.h"

void test(int B, int CI, int CO, int HW) {
    float *output = new float[B * CO * HW];
    float *input = new float[B * CI * HW];
    float *input2 = new float[B * CI * HW];
    float *weight = new float[CO * CI];
    for (int i = 0; i < B * CI * HW; i++) {
        float mid = ((float)rand() - 0.5) / RAND_MAX;
        float r = ((float)rand()) / RAND_MAX;
        input[i] = mid - r;
        input2[i] = mid + r;
    }
    for (int i = 0; i < CO * CI; i++)
        weight[i] = ((float)rand() - 0.5) / RAND_MAX;
    float *input_cuda, *input2_cuda, *weight_cuda, *output_cuda, *output2_cuda;
    cudaMallocManaged(&output_cuda, B * CO * HW * sizeof(float));
    cudaMallocManaged(&output2_cuda, B * CO * HW * sizeof(float));
    cudaMallocManaged(&input_cuda, B * CI * HW * sizeof(float));
    cudaMallocManaged(&input2_cuda, B * CI * HW * sizeof(float));
    cudaMallocManaged(&weight_cuda, CO * CI * sizeof(float));

    cudaMemcpy(input_cuda, input, B * CI * HW * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(input2_cuda, input2, B * CI * HW * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_cuda, weight, CO * CI * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 1; j++)
            bound_sort_neuron_forward_cuda(input_cuda, input2_cuda, weight_cuda, B, CO, CI, 1, HW, output_cuda, output2_cuda, 0.6, 25);
//            sort_neuron_forward_cuda(input_cuda, weight_cuda, B, CO, CI, 1, HW, output_cuda, 0.6, 25);
    }
    cudaDeviceSynchronize();

    cudaFree(output_cuda);
    cudaFree(output2_cuda);
    cudaFree(input_cuda);
    cudaFree(input2_cuda);
    cudaFree(weight_cuda);
    delete[] output;
    delete[] input;
    delete[] input2;
    delete[] weight;
}

int main() {
//    test(512, 128 * 3 * 3, 128, 16 * 16);
    test(512, 5120, 5120, 1);
//    test(512, 32768, 512, 1);
}