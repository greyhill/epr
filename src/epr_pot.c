#include "epr.h"

float eprPotential_eval(struct eprPotential* e, float v) {
    if(e->eval_fn != NULL) {
        return (e->eval_fn)(e, v);
    } else {
        return NAN;
    }
}

float eprPotential_grad(struct eprPotential* e, float v) {
    if(e->grad_fn != NULL) {
        return (e->grad_fn)(e, v);
    } else {
        return NAN;
    }
}

float eprPotential_huber(struct eprPotential* e, float v) {
    if(e->huber_fn != NULL) {
        return (e->huber_fn)(e, v);
    } else {
        return NAN;
    }
}
