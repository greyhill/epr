#include "epr.h"

size_t compute_np(size_t nd, size_t* dims) {
    size_t tr = 1;
    for(size_t id=0; id<nd; ++id) {
        tr *= dims[id];
    }
    return tr;
}

bool compute_offset(size_t nd, size_t* dims, int32_t* diffs,
                    size_t ip,
                    bool reverse, size_t* ip_out) {
    size_t out_accum = 0;
    size_t stride_accum = 1;
    for(size_t id=0; id<nd; ++id) {
        size_t idim = dims[id];
        int32_t idiff = diffs[id];

        int64_t ipd = ip % idim;
        int64_t ipd1 = 0;
        if(reverse) {
            ipd1 = ipd - idiff;
        } else {
            ipd1 = ipd + idiff;
        }
        if(ipd1 < 0 || ipd1 >= (int64_t)idim) {
            return false;
        }

        out_accum += stride_accum * ipd1;
        ip /= idim;
        stride_accum *= idim;
    }

    *ip_out = out_accum;
    return true;
}

float eprImage_eval(size_t nd,
                    size_t* dims,
                    size_t npot,
                    int32_t* diffs,
                    struct eprPotential** diff_pots,
                    float* weights,
                    const float* image) {
    double accum = 0.0;
    size_t np = compute_np(nd, dims);

#pragma omp parallel for reduction(+:accum)
    for(size_t ip=0; ip<np; ++ip) {
        float pixel_accum = 0.0;

        float xip = image[ip];
        float wip = 1.f;
        if(weights != NULL) {
            wip = weights[ip];
        }

        for(size_t ipot=0; ipot<npot; ++ipot) {
            size_t ip1=0;
            struct eprPotential* pot = diff_pots[ipot];

            if(compute_offset(nd, dims, diffs+nd*ipot, ip, false, &ip1)) {
                float xip1 = image[ip1];
                float wip1 = 1.f;
                if(weights != NULL) {
                    wip1 = weights[ip1];
                }
                pixel_accum += wip*wip1* (pot->eval_fn)(pot, xip - xip1);
            }
        }

        accum += pixel_accum;
    }

    return accum;
}

void eprImage_grad(size_t nd,
                   size_t* dims,
                   size_t npot,
                   int32_t* diffs,
                   struct eprPotential** diff_pots,
                   float* weights,
                   const float* image,
                   float* out) {
    size_t np = compute_np(nd, dims);

#pragma omp parallel for
    for(size_t ip=0; ip<np; ++ip) {
        float pixel_accum = 0.0;

        float xip = image[ip];
        float wip = 1.f;
        if(weights != NULL) {
            wip = weights[ip];
        }

        for(size_t ipot=0; ipot<npot; ++ipot) {
            size_t ip1=0;
            struct eprPotential* pot = diff_pots[ipot];
            if(pot->grad_fn == NULL) {
                continue;
            }

            if(compute_offset(nd, dims, diffs+nd*ipot, ip, false, &ip1)) {
                float xip1 = image[ip1];
                float wip1 = 1.f;
                if(weights != NULL) {
                    wip1 = weights[ip1];
                }
                pixel_accum += wip*wip1* (pot->grad_fn)(pot, xip - xip1);
            }

            if(compute_offset(nd, dims, diffs+nd*ipot, ip, true, &ip1)) {
                float xip1 = image[ip1];
                float wip1 = 1.f;
                if(weights != NULL) {
                    wip1 = weights[ip1];
                }
                pixel_accum += wip*wip1* (pot->grad_fn)(pot, xip - xip1);
            }
        }

        out[ip] = pixel_accum;
    }
}

void eprImage_huber(size_t nd,
                    size_t* dims,
                    size_t npot,
                    int32_t* diffs,
                    struct eprPotential** diff_pots,
                    float* weights,
                    const float* image,
                    float* out) {
    size_t np = compute_np(nd, dims);

#pragma omp parallel for
    for(size_t ip=0; ip<np; ++ip) {
        float pixel_accum = 0.0;

        float xip = image[ip];
        float wip = 1.f;
        if(weights != NULL) {
            wip = weights[ip];
        }

        for(size_t ipot=0; ipot<npot; ++ipot) {
            size_t ip1 = 0;
            struct eprPotential* pot = diff_pots[ipot];
            if(pot->huber_fn == NULL) continue;

            if(compute_offset(nd, dims, diffs+nd*ipot, ip, false, &ip1)) {
                float xip1 = image[ip1];
                float wip1 = 1.f;
                if(weights != NULL) {
                    wip1 = weights[ip1];
                }
                pixel_accum += wip*wip1* (pot->huber_fn)(pot, xip - xip1);
            }

            if(compute_offset(nd, dims, diffs+nd*ipot, ip, true, &ip1)) {
                float xip1 = image[ip1];
                float wip1 = 1.f;
                if(weights != NULL) {
                    wip1 = weights[ip1];
                }
                pixel_accum += wip*wip1* (pot->huber_fn)(pot, xip - xip1);
            }
        }

        out[ip] = pixel_accum;
    }
}

