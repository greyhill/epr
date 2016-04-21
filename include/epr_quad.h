struct eprQuadratic {
    float (*eval_fn)(struct eprQuadratic*, float);
    float (*grad_fn)(struct eprQuadratic*, float);
    float (*huber_fn)(struct eprQuadratic*, float);
    float beta;
};

extern void eprQuadratic_init(struct eprQuadratic* q);

extern float eprQuadratic_eval(struct eprQuadratic* q, float d);
extern float eprQuadratic_grad(struct eprQuadratic* q, float d);
extern float eprQuadratic_huber(struct eprQuadratic* q, float d);

