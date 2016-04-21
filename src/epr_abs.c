#include "epr.h"

void eprAbs_init(struct eprAbs* a) {
    a->eval_fn = &eprAbs_eval;
    a->grad_fn = NULL;
    a->huber_fn = NULL;
    a->beta = 0.f;
}

float eprAbs_eval(struct eprAbs* a, float d) {
    return a->beta * fabsf(d);
}
