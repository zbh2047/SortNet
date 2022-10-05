#include "norm_dist.h"

template <int p>
void test(int B, int CI, int CO, int HW, float fp) {
    float *output = new float[B * CO * HW];
    float *input = new float[B * CI * HW];
    float *weight = new float[CO * CI];
    float *grad_output = new float[B * CO * HW];
    for (int i = 0; i < B * CI * HW; i++)
        input[i] = ((float)rand() - 0.5) / RAND_MAX;
    for (int i = 0; i < CO * CI; i++)
        weight[i] = ((float)rand() - 0.5) / RAND_MAX;
    for (int i = 0; i < B * CO * HW; i++)
        grad_output[i] = ((float)rand() - 0.5) / RAND_MAX;
    float *input_cuda, *weight_cuda, *output_cuda, *grad_output_cuda, *grad_input_cuda, *grad_weight_cuda;
    int *pos_cuda;
    cudaMallocManaged(&output_cuda, B * CO * HW * sizeof(float));
    cudaMallocManaged(&pos_cuda, B * CO * HW * sizeof(int));
    cudaMallocManaged(&input_cuda, B * CI * HW * sizeof(float));
    cudaMallocManaged(&weight_cuda, CO * CI * sizeof(float));
    cudaMallocManaged(&grad_output_cuda, B * CO * HW * sizeof(float));
    cudaMallocManaged(&grad_input_cuda, B * CI * HW * sizeof(float));
    cudaMallocManaged(&grad_weight_cuda, CO * CI * sizeof(float));

    cudaMemcpy(input_cuda, input, B * CI * HW * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_cuda, weight, CO * CI * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_output_cuda, grad_output, B * CI * HW * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 1; j++)
            norm_dist<p>::forward_with_max(input_cuda, weight_cuda, nullptr, B, CO, CI, 1, HW, output_cuda, fp, nullptr, nullptr, CO, CI);
        for (int j = 0; j < 1; j++)
            norm_dist<p>::forward(input_cuda, weight_cuda, B, CO, CI, 1, HW, output_cuda, fp, nullptr, nullptr, CO, CI);
//        for (int j = 0; j < 1; j++)
//            norm_dist<p >= 0 ? p - 1 : p>::backward_input(grad_output_cuda, input_cuda, weight_cuda, output_cuda, B, CO, CI, 1, HW, grad_input_cuda, fp, nullptr, nullptr, CO, CI);
        for (int j = 0; j < 1; j++)
            norm_dist<p >= 0 ? p - 1 : p>::backward_input_weight(grad_output_cuda, input_cuda, weight_cuda, output_cuda, B, CO, CI, 1, HW, grad_input_cuda, grad_weight_cuda, fp, nullptr, nullptr, CO, CI);
    }
    cudaDeviceSynchronize();

    cudaFree(output_cuda);
    cudaFree(pos_cuda);
    cudaFree(input_cuda);
    cudaFree(weight_cuda);
    cudaFree(grad_input_cuda);
    cudaFree(grad_weight_cuda);
    cudaFree(grad_output_cuda);
    delete[] output;
    delete[] input;
    delete[] weight;
    delete[] grad_output;
}

int main() {
//    test<8>(512, 128 * 3 * 3, 128, 16 * 16, 8);
//    test<-1>(512, 128 * 3 * 3, 128, 16 * 16, 20);
    test<8>(512, 5120, 5120, 1, 8);
    test<-1>(512, 5120, 5120, 1, 20);
//    test<8>(512, 32768, 512, 1, 8);
//    test<-1>(512, 32768, 512, 1, 20);
}