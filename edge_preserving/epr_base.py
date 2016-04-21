import ctypes as ct
import ctypes.util
import numpy as np
import itertools

libpath = ctypes.util.find_library('libepr')
if libpath is None:
    import os
    import os.path
    libpath = os.path.expanduser("~/lib/libepr.so")
lib = ct.CDLL(libpath)

class Potential(object):
    def eval(self, d):
        lib.eprPotential_eval.restype = ct.c_float
        return lib.eprPotential_eval(ct.pointer(self), ct.c_float(d))

    def grad(self, d):
        lib.eprPotential_grad.restype = ct.c_float
        return lib.eprPotential_grad(ct.pointer(self), ct.c_float(d))

    def huber(self, d):
        lib.eprPotential_huber.restype = ct.c_float
        return lib.eprPotential_huber(ct.pointer(self), ct.c_float(d))

    def grad_image(self, x, diffs, weights=None):
        tr = np.zeros(x.shape, dtype='float32', order='f')
        x = np.asarray(x, dtype='float32', order='f')
        w = weights
        if w is not None:
            w = np.asarray(w, dtype='float32', order='f')
        diffs = np.asarray(diffs, dtype='int32', order='f')
        ndiff = len(diffs) / len(tr.shape)
        diff_ptrs = (ct.c_voidp * ndiff)(*((\
                ct.cast(ct.pointer(self), ct.c_voidp),)*ndiff))
        dims = np.asarray(tr.shape, dtype=ct.c_size_t)
        lib.eprImage_grad(\
                ct.c_size_t(len(tr.shape)),
                ct.c_voidp(dims.ctypes.data),
                ct.c_size_t(ndiff),
                ct.c_voidp(diffs.ctypes.data),
                ct.pointer(diff_ptrs),
                ct.c_voidp(0) if w is None else ct.c_voidp(w.ctypes.data),
                ct.c_voidp(x.ctypes.data),
                ct.c_voidp(tr.ctypes.data))
        return tr

    def huber_image(self, x, diffs, weights=None):
        tr = np.zeros(x.shape, dtype='float32', order='f')
        x = np.asarray(x, dtype='float32', order='f')
        w = weights
        if w is not None:
            w = np.asarray(w, dtype='float32', order='f')
        diffs = np.asarray(diffs, dtype='int32', order='f')
        ndiff = len(diffs) / len(tr.shape)
        diff_ptrs = (ct.c_voidp * ndiff)(*((\
                ct.cast(ct.pointer(self), ct.c_voidp),)*ndiff))
        dims = np.asarray(tr.shape, dtype=ct.c_size_t)
        lib.eprImage_huber(\
                ct.c_size_t(len(tr.shape)),
                ct.c_voidp(dims.ctypes.data),
                ct.c_size_t(ndiff),
                ct.c_voidp(diffs.ctypes.data),
                ct.pointer(diff_ptrs),
                ct.c_voidp(0) if w is None else ct.c_voidp(w.ctypes.data),
                ct.c_voidp(x.ctypes.data),
                ct.c_voidp(tr.ctypes.data))
        return tr

class Quadratic(ct.Structure, Potential):
    _fields_ = [ \
            ('eval_fn', ct.c_voidp),
            ('grad_fn', ct.c_voidp),
            ('huber_fn', ct.c_voidp),
            ('beta', ct.c_float),
        ]
    def __init__(self, beta=0):
        lib.eprQuadratic_init(ct.pointer(self))
        self.beta = beta

class Abs(ct.Structure, Potential):
    _fields_ = [ \
            ('eval_fn', ct.c_voidp),
            ('grad_fn', ct.c_voidp),
            ('huber_fn', ct.c_voidp),
            ('beta', ct.c_float),
        ]
    def __init__(self, beta=0):
        lib.eprAbs_init(ct.pointer(self))
        self.beta = beta

