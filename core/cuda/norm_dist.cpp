#include <torch/extension.h>
#include "norm_dist.h"

#define MIN_NORMALIZED_P 10.0

typedef torch::Tensor Tensor;
typedef at::optional<torch::Tensor> OptTensor;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_OP_INPUT(x) if (x.has_value()) { CHECK_INPUT(x.value()); }

#define GET_PTR(tensor) (tensor).data_ptr<float>()
#define GET_OP_INT_PTR(tensor) ((tensor).has_value() ? (tensor)->data_ptr<int>() : nullptr)

#define call_intp(type, fun, p, paras...) {\
    type<p>::fun(paras); \
}
#define call_p(type, fun, p, paras...) { \
    static_assert(MIN_INSTANTIATED_P == 0, "p out of range"); \
    static_assert(MAX_INSTANTIATED_P == 8, "p out of range"); \
    if ((p) == float(int(p))) { \
        switch (int(p)) { \
            case 0: call_intp(type, fun, 0, paras); break; \
            case 1: call_intp(type, fun, 1, paras); break; \
            case 2: call_intp(type, fun, 2, paras); break; \
            case 3: call_intp(type, fun, 3, paras); break; \
            case 4: call_intp(type, fun, 4, paras); break; \
            case 5: call_intp(type, fun, 5, paras); break; \
            case 6: call_intp(type, fun, 6, paras); break; \
            case 7: call_intp(type, fun, 7, paras); break; \
            case 8: call_intp(type, fun, 8, paras); break; \
            default: call_intp(type, fun, -1, paras); \
        } \
    } \
    else call_intp(type, fun, -1, paras); \
}
#define call(type, fun, paras...) { \
    type::fun(paras); \
}

void norm_dist_forward(const Tensor& input, const Tensor& weight, Tensor& output, int G, float p,
                       const OptTensor& w_index_ci, const OptTensor& w_index_co) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    CHECK_OP_INPUT(w_index_ci);
    CHECK_OP_INPUT(w_index_co);
    int B = input.size(0), CO = output.size(1), CI = input.size(1), HW = input.size(2);
    if (p <= MIN_NORMALIZED_P) {
        call_p(norm_dist, forward_with_max, p,
               GET_PTR(input), GET_PTR(weight), nullptr, B, CO, CI, G, HW, GET_PTR(output), p,
               GET_OP_INT_PTR(w_index_ci), GET_OP_INT_PTR(w_index_co), weight.size(0), weight.size(1) * G);
    }
    else {
        call_p(norm_dist, forward, p,
               GET_PTR(input), GET_PTR(weight), B, CO, CI, G, HW, GET_PTR(output), p,
               GET_OP_INT_PTR(w_index_ci), GET_OP_INT_PTR(w_index_co), weight.size(0), weight.size(1) * G);
    }
}

void norm_dist_forward_(const Tensor& input, const Tensor& weight, Tensor& output, int G, float p) {
    norm_dist_forward(input, weight, output, G, p, {}, {});
}

void norm_dist_forward_with_max(const Tensor& input, const Tensor& weight, const Tensor& max_output, Tensor& output,
                                int G, float p, const OptTensor& w_index_ci, const OptTensor& w_index_co) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(max_output);
    CHECK_INPUT(output);
    CHECK_OP_INPUT(w_index_ci);
    CHECK_OP_INPUT(w_index_co);
    int B = input.size(0), CO = output.size(1), CI = input.size(1), HW = input.size(2);
    call_p(norm_dist, forward_with_max, p,
           GET_PTR(input), GET_PTR(weight), GET_PTR(max_output), B, CO, CI, G, HW, GET_PTR(output), p,
           GET_OP_INT_PTR(w_index_ci), GET_OP_INT_PTR(w_index_co), weight.size(0), weight.size(1) * G);
}

void norm_dist_forward_with_max_(const Tensor& input, const Tensor& weight, const Tensor& max_output, Tensor& output,
                                 int G, float p) {
    norm_dist_forward_with_max(input, weight, max_output, output, G, p, {}, {});
}

void norm_dist_backward_input(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const Tensor& output,
                              Tensor& grad_input, int G, float p,
                              const OptTensor& w_index_ci, const OptTensor& w_index_co) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    CHECK_INPUT(grad_input);
    CHECK_OP_INPUT(w_index_ci);
    CHECK_OP_INPUT(w_index_co);
    int B = grad_output.size(0), CO = output.size(1), CI = input.size(1), HW = input.size(2);
    call_p(norm_dist, backward_input, p - 1,
           GET_PTR(grad_output), GET_PTR(input), GET_PTR(weight), GET_PTR(output), B, CO, CI, G, HW,
           GET_PTR(grad_input), p,
           GET_OP_INT_PTR(w_index_ci), GET_OP_INT_PTR(w_index_co), weight.size(0), weight.size(1) * G);
}

void norm_dist_backward_input_(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const Tensor& output,
                               Tensor& grad_input, int G, float p) {
    norm_dist_backward_input(grad_output, input, weight, output, grad_input, G, p, {}, {});
}

void norm_dist_backward_input_weight(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                                     const Tensor& output, Tensor& grad_input, Tensor& grad_weight, int G, float p,
                                     const OptTensor& w_index_ci, const OptTensor& w_index_co) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    CHECK_INPUT(grad_input);
    CHECK_INPUT(grad_weight);
    CHECK_OP_INPUT(w_index_ci);
    CHECK_OP_INPUT(w_index_co);
    int B = grad_output.size(0), CO = output.size(1), CI = input.size(1), HW = input.size(2);
    call_p(norm_dist, backward_input_weight, p - 1,
           GET_PTR(grad_output), GET_PTR(input), GET_PTR(weight), GET_PTR(output), B, CO, CI, G, HW,
           GET_PTR(grad_input), GET_PTR(grad_weight), p,
           GET_OP_INT_PTR(w_index_ci), GET_OP_INT_PTR(w_index_co), weight.size(0), weight.size(1) * G);
}

void norm_dist_backward_input_weight_(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                                      const Tensor& output, Tensor& grad_input, Tensor& grad_weight, int G, float p) {
    norm_dist_backward_input_weight(grad_output, input, weight, output, grad_input, grad_weight, G, p, {}, {});
}

void inf_dist_forward(const Tensor& input, const Tensor& weight, Tensor& output, Tensor& pos, int G,
                      const OptTensor& w_index_ci, const OptTensor& w_index_co) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    CHECK_INPUT(pos);
    CHECK_OP_INPUT(w_index_ci);
    CHECK_OP_INPUT(w_index_co);
    int B = input.size(0), CO = output.size(1), CI = input.size(1), HW = input.size(2);
    call(inf_dist, forward,
         GET_PTR(input), GET_PTR(weight), B, CO, CI, G, HW, GET_PTR(output), pos.data_ptr<int>(),
         GET_OP_INT_PTR(w_index_ci), GET_OP_INT_PTR(w_index_co), weight.size(0), weight.size(1) * G);
}

void inf_dist_forward_(const Tensor& input, const Tensor& weight, Tensor& output, Tensor& pos, int G) {
    inf_dist_forward(input, weight, output, pos, G, {}, {});
}

void inf_dist_forward_nograd(const Tensor& input, const Tensor& weight, Tensor& output, int G) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    int B = input.size(0), CO = output.size(1), CI = input.size(1), HW = input.size(2);
    call(inf_dist, forward_nograd,
         GET_PTR(input), GET_PTR(weight), B, CO, CI, G, HW, GET_PTR(output));
}

void inf_dist_backward_input(const Tensor& grad_output, const Tensor& pos, Tensor& grad_input, int G) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(pos);
    CHECK_INPUT(grad_input);
    int B = grad_output.size(0), CO = grad_output.size(1), CI = grad_input.size(1), HW = grad_input.size(2);
    call(inf_dist, backward_input,
         GET_PTR(grad_output), pos.data_ptr<int>(), B, CO, CI, G, HW, GET_PTR(grad_input));
}

void inf_dist_backward_input_weight(const Tensor& grad_output, const Tensor& pos, Tensor& grad_input, Tensor& grad_weight,
                                    int G, const OptTensor& w_index_ci, const OptTensor& w_index_co) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(pos);
    CHECK_INPUT(grad_input);
    CHECK_INPUT(grad_weight);
    CHECK_OP_INPUT(w_index_ci);
    CHECK_OP_INPUT(w_index_co);
    int B = grad_output.size(0), CO = grad_output.size(1), CI = grad_input.size(1), HW = grad_output.size(2);
    call(inf_dist, backward_input_weight,
         GET_PTR(grad_output), pos.data_ptr<int>(), B, CO, CI, G, HW, GET_PTR(grad_input), GET_PTR(grad_weight),
         GET_OP_INT_PTR(w_index_ci), GET_OP_INT_PTR(w_index_co), grad_weight.size(0), grad_weight.size(1) * G);
}

void inf_dist_backward_input_weight_(const Tensor& grad_output, const Tensor& pos, Tensor& grad_input,
                                     Tensor& grad_weight, int G) {
    inf_dist_backward_input_weight(grad_output, pos, grad_input, grad_weight, G, {}, {});
}

void bound_inf_dist_forward_nograd(const Tensor& inputL, const Tensor& inputU, const Tensor& weight,
                                   Tensor& outputL, Tensor& outputU, int G) {
    CHECK_INPUT(inputL);
    CHECK_INPUT(inputU);
    CHECK_INPUT(weight);
    CHECK_INPUT(outputL);
    CHECK_INPUT(outputU);
    int B = inputL.size(0), CO = outputL.size(1), CI = inputL.size(1), HW = inputL.size(2);
    call(bound_inf_dist, forward_nograd,
         GET_PTR(inputL), GET_PTR(inputU), GET_PTR(weight), B, CO, CI, G, HW, GET_PTR(outputL), GET_PTR(outputU));
}

void bound_inf_dist_forward(const Tensor& inputL, const Tensor& inputU, const Tensor& weight,
                            Tensor& outputL, Tensor& outputU, Tensor& posL, Tensor& posU, int G) {
    CHECK_INPUT(inputL);
    CHECK_INPUT(inputU);
    CHECK_INPUT(weight);
    CHECK_INPUT(outputL);
    CHECK_INPUT(outputU);
    CHECK_INPUT(posL);
    CHECK_INPUT(posU);
    int B = inputL.size(0), CO = outputL.size(1), CI = inputL.size(1), HW = inputL.size(2);
    call(bound_inf_dist, forward,
         GET_PTR(inputL), GET_PTR(inputU), GET_PTR(weight), B, CO, CI, G, HW,
         GET_PTR(outputL), GET_PTR(outputU), posL.data_ptr<int>(), posU.data_ptr<int>());
}

void bound_inf_dist_backward_input(const Tensor& grad_outputL, const Tensor& grad_outputU, const Tensor& posL,
                                   const Tensor& posU, Tensor& grad_inputL, Tensor& grad_inputU, int G) {
    CHECK_INPUT(grad_outputL);
    CHECK_INPUT(grad_outputU);
    CHECK_INPUT(posL);
    CHECK_INPUT(posU);
    CHECK_INPUT(grad_inputL);
    CHECK_INPUT(grad_inputU);
    int B = grad_outputL.size(0), CO = grad_outputL.size(1), CI = grad_inputL.size(1), HW = grad_inputL.size(2);
    call(bound_inf_dist, backward_input,
         GET_PTR(grad_outputL), GET_PTR(grad_outputU), posL.data_ptr<int>(), posU.data_ptr<int>(),
         B, CO, CI, G, HW, GET_PTR(grad_inputL), GET_PTR(grad_inputU));
}

void bound_inf_dist_backward_input_weight(const Tensor& grad_outputL, const Tensor& grad_outputU, const Tensor& posL,
                                          const Tensor& posU, Tensor& grad_inputL, Tensor& grad_inputU, Tensor& grad_weight,
                                          int G) {
    CHECK_INPUT(grad_outputL);
    CHECK_INPUT(grad_outputU);
    CHECK_INPUT(posL);
    CHECK_INPUT(posU);
    CHECK_INPUT(grad_inputL);
    CHECK_INPUT(grad_inputU);
    CHECK_INPUT(grad_weight);
    int B = grad_outputL.size(0), CO = grad_outputL.size(1), CI = grad_weight.size(1) * G, HW = grad_outputL.size(2);
    call(bound_inf_dist, backward_input_weight,
         GET_PTR(grad_outputL), GET_PTR(grad_outputU), posL.data_ptr<int>(), posU.data_ptr<int>(),
         B, CO, CI, G, HW, GET_PTR(grad_inputL), GET_PTR(grad_inputU), GET_PTR(grad_weight));
}

void sort_neuron_forward(const Tensor& input, const Tensor& weight, Tensor& output, Tensor& pos, int G, float q, int K) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    CHECK_INPUT(pos);
    int B = input.size(0), CO = output.size(1), CI = input.size(1), HW = input.size(2);
    call(sort_neuron, forward,
         GET_PTR(input), GET_PTR(weight), B, CO, CI, G, HW, GET_PTR(output), pos.data_ptr<int>(), q, K);
}

void bound_sort_neuron_forward(const Tensor& inputL, const Tensor& inputU, const Tensor& weight,
                               Tensor& outputL, Tensor& outputU, int G, float q, int K) {
    CHECK_INPUT(inputL);
    CHECK_INPUT(inputU);
    CHECK_INPUT(weight);
    CHECK_INPUT(outputL);
    CHECK_INPUT(outputU);
    int B = inputL.size(0), CO = outputL.size(1), CI = inputL.size(1), HW = inputL.size(2);
    bound_sort_neuron_forward_cuda(GET_PTR(inputL), GET_PTR(inputU), GET_PTR(weight), B, CO, CI, G, HW,
                                   GET_PTR(outputL), GET_PTR(outputU), q, K);
}

void sort_neuron_forward_nograd(const Tensor& input, const Tensor& weight, Tensor& output, int G, float q, int K) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    int B = input.size(0), CO = output.size(1), CI = input.size(1), HW = input.size(2);
    call(sort_neuron, forward_nograd,
         GET_PTR(input), GET_PTR(weight), B, CO, CI, G, HW, GET_PTR(output), q, K);
}

void sort_neuron_backward_input(const Tensor& grad_output, const Tensor& pos, Tensor& grad_input, int G, float q, int K) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(pos);
    CHECK_INPUT(grad_input);
    int B = grad_output.size(0), CO = grad_output.size(1), CI = grad_input.size(1), HW = grad_input.size(2);
    call(sort_neuron, backward_input,
         GET_PTR(grad_output), pos.data_ptr<int>(), B, CO, CI, G, HW, GET_PTR(grad_input), q, K);
}

void sort_neuron_backward_input_weight(const Tensor& grad_output, const Tensor& pos, Tensor& grad_input,
                                       Tensor& grad_weight, int G, float q, int K) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(pos);
    CHECK_INPUT(grad_input);
    CHECK_INPUT(grad_weight);
    int B = grad_output.size(0), CO = grad_output.size(1), CI = grad_input.size(1), HW = grad_output.size(2);
    call(sort_neuron, backward_input_weight,
         GET_PTR(grad_output), pos.data_ptr<int>(), B, CO, CI, G, HW, GET_PTR(grad_input), GET_PTR(grad_weight), q, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("norm_dist_forward", &norm_dist_forward);
    m.def("norm_dist_forward", &norm_dist_forward_);
    m.def("norm_dist_forward_with_max", &norm_dist_forward_with_max);
    m.def("norm_dist_forward_with_max", &norm_dist_forward_with_max_);
    m.def("norm_dist_backward_input", &norm_dist_backward_input);
    m.def("norm_dist_backward_input", &norm_dist_backward_input_);
    m.def("norm_dist_backward_input_weight", &norm_dist_backward_input_weight);
    m.def("norm_dist_backward_input_weight", &norm_dist_backward_input_weight_);
    m.def("inf_dist_forward", &inf_dist_forward);
    m.def("inf_dist_forward", &inf_dist_forward_);
    m.def("inf_dist_forward_nograd", &inf_dist_forward_nograd);
    m.def("inf_dist_backward_input", &inf_dist_backward_input);
    m.def("inf_dist_backward_input_weight", &inf_dist_backward_input_weight);
    m.def("inf_dist_backward_input_weight", &inf_dist_backward_input_weight_);
    m.def("bound_inf_dist_forward", &bound_inf_dist_forward);
    m.def("bound_inf_dist_forward_nograd", &bound_inf_dist_forward_nograd);
    m.def("bound_inf_dist_backward_input", &bound_inf_dist_backward_input);
    m.def("bound_inf_dist_backward_input_weight", &bound_inf_dist_backward_input_weight);
    m.def("sort_neuron_forward", &sort_neuron_forward);
    m.def("bound_sort_neuron_forward", &bound_sort_neuron_forward);
    m.def("sort_neuron_forward_nograd", &sort_neuron_forward_nograd);
    m.def("sort_neuron_backward_input", &sort_neuron_backward_input);
    m.def("sort_neuron_backward_input_weight", &sort_neuron_backward_input_weight);
}