struct eprPotential {
    float (*eval_fn)(struct eprPotential*, float);
    float (*grad_fn)(struct eprPotential*, float);
    float (*huber_fn)(struct eprPotential*, float);
};

extern float eprPotential_eval(struct eprPotential* e, float d);
extern float eprPotential_grad(struct eprPotential* e, float d);
extern float eprPotential_huber(struct eprPotential* e, float d);

