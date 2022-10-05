#define MIN_INSTANTIATED_P 0
#define MAX_INSTANTIATED_P 8

template <int ip> struct norm_dist {
    static void forward(const float* input, const float* weight,
                        int B, int CO, int CI, int G, int HW, float* output, float p,
                        const int* w_index_ci, const int* w_index_co, int W_CO, int W_CI);
    static void forward_with_max(const float* input, const float* weight, const float* max_output,
                                 int B, int CO, int CI, int G, int HW, float* output, float p,
                                 const int* w_index_ci, const int* w_index_co, int W_CO, int W_CI);
    static void backward_input(const float* grad_output, const float* input, const float* weight, const float* output,
                               int B, int CO, int CI, int G, int HW, float* grad_input, float p,
                               const int* w_index_ci, const int* w_index_co, int W_CO, int W_CI);
    static void backward_input_weight(const float* grad_output, const float* input, const float* weight,
                                      const float* output, int B, int CO, int CI, int G, int HW,
                                      float* grad_input, float* grad_weight, float p,
                                      const int* w_index_ci, const int* w_index_co, int W_CO, int W_CI);
};

struct inf_dist {
    static void forward_nograd(const float* input, const float* weight,
                               int B, int CO, int CI, int G, int HW, float* output);
    static void forward(const float* input, const float* weight,
                        int B, int CO, int CI, int G, int HW, float* output, int* pos,
                        const int* w_index_ci, const int* w_index_co, int W_CO, int W_CI);
    static void backward_input(const float* grad_output, const int* pos,
                               int B, int CO, int CI, int G, int HW, float* grad_input);
    static void backward_input_weight(const float* grad_output, const int* pos,
                                      int B, int CO, int CI, int G, int HW, float* grad_input, float* grad_weight,
                                      const int* w_index_ci, const int* w_index_co, int W_CO, int W_CI);
};

struct bound_inf_dist {
    static void forward_nograd(const float* inputL, const float* inputU, const float* weight,
                               int B, int CO, int CI, int G, int HW, float* outputL, float* outputU);
    static void forward(const float* inputL, const float* inputU, const float* weight,
                        int B, int CO, int CI, int G, int HW, float* outputL, float* outputU, int* posL, int* posU);
    static void backward_input(const float* grad_outputL, const float* grad_outputU,
                               const int* posL, const int* posU, int B, int CO, int CI, int G, int HW,
                               float* grad_inputL, float* grad_inputU);
    static void backward_input_weight(const float* grad_outputL, const float* grad_outputU,
                                      const int* posL, const int* posU, int B, int CO, int CI, int G, int HW,
                                      float* grad_inputL, float* grad_inputU, float* grad_weight);
};

struct sort_neuron {
    static void forward_nograd(const float* input, const float* weight,
                               int B, int CO, int CI, int G, int HW, float* output, float q, int K);
    static void forward(const float* input, const float* weight,
                        int B, int CO, int CI, int G, int HW, float* output, int* pos, float q, int K);
    static void backward_input(const float* grad_output, const int* pos,
                               int B, int CO, int CI, int G, int HW, float* grad_input, float q, int K);
    static void backward_input_weight(const float* grad_output, const int* pos,
                                      int B, int CO, int CI, int G, int HW, float* grad_input, float* grad_weight,
                                      float q, int K);
};


void bound_sort_neuron_forward_cuda(const float* inputL, const float* inputU, const float* weight,
                                    int B, int CO, int CI, int G, int HW, float* outputL, float* outputU, float q, int K);