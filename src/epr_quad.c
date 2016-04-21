#include "epr.h"

void eprQuadratic_init(struct eprQuadratic* q) {
    q->eval_fn = &eprQuadratic_eval;
    q->grad_fn = &eprQuadratic_grad;
    q->huber_fn = &eprQuadratic_huber;
    q->beta = 0.f;
}

float eprQuadratic_eval(struct eprQuadratic* q, float d) {
    return q->beta/2.f * d*d;
}

float eprQuadratic_grad(struct eprQuadratic* q, float d) {
    return q->beta * d;
}

float eprQuadratic_huber(struct eprQuadratic* q, float d) {
    (void)d;
    return q->beta;
}

