struct eprAbs {
    float (*eval_fn)(struct eprAbs*, float);
    float (*grad_fn)(struct eprAbs*, float);
    float (*huber_fn)(struct eprAbs*, float);
    float beta;
};

extern void eprAbs_init(struct eprAbs* a);
extern float eprAbs_eval(struct eprAbs* a, float d);

