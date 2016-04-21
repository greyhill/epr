extern float eprImage_eval(size_t nd,
                           size_t* dims,
                           size_t npot,
                           int32_t* diffs,
                           struct eprPotential** diff_pots,
                           float* weights,
                           const float* image);

extern void eprImage_grad(size_t nd,
                          size_t* dims,
                          size_t npot,
                          int32_t* diffs,
                          struct eprPotential** diff_pots,
                          float* weights,
                          const float* image,
                          float* out);

extern void eprImage_huber(size_t nd,
                           size_t* dims,
                           size_t npot,
                           int32_t* diffs,
                           struct eprPotential** diff_pots,
                           float* weights,
                           const float* image,
                           float* out);

