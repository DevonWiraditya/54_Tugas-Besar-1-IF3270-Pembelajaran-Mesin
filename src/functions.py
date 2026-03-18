import numpy as np
from autodiff import Tensor

def zero_init(shape):
    return Tensor(np.zeros(shape))

def uniform_init(shape, low, high, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return Tensor(np.random.uniform(low, high, size=shape))

def normal_init(shape, mean, variance, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return Tensor(np.random.normal(mean, np.sqrt(variance), size=shape))

def linear(x):
    return x

def relu(x):
    return x.maximum(0.0)

def sigmoid(x):
    return x.sigmoid()

def tanh(x):
    return x.tanh()

def softmax(x):
    max_vals = Tensor(np.max(x.data, axis=1, keepdims=True), requires_grad=False)
    exps = (x - max_vals).exp()
    sums = exps.sum(axis=1, keepdims=True)
    return exps / sums

def mse(y_pred, y_true):
    yt = y_true if isinstance(y_true, Tensor) else Tensor(y_true, requires_grad=False)
    return ((y_pred - yt) ** 2).mean()

def binary_crossentropy(y_pred, y_true):
    yt = y_true if isinstance(y_true, Tensor) else Tensor(y_true, requires_grad=False)
    y_pred = y_pred.clip(1e-15, 1.0 - 1e-15)
    
    loss_sum = (yt * y_pred.log() + (Tensor(1.0, requires_grad=False) - yt) * (Tensor(1.0, requires_grad=False) - y_pred).log()).sum()
    N = yt.data.shape[0] if len(yt.data.shape) > 0 else 1
    return -loss_sum / N

def categorical_crossentropy(y_pred, y_true):
    yt = y_true if isinstance(y_true, Tensor) else Tensor(y_true, requires_grad=False)
    y_pred = y_pred.clip(1e-15, 1.0 - 1e-15)
    
    loss_sum = (yt * y_pred.log()).sum()
    N = yt.data.shape[0] if len(yt.data.shape) > 0 else 1
    return -loss_sum / N

def get_activation(name):
    name = name.lower()
    if name == 'relu': return relu
    if name == 'sigmoid': return sigmoid
    if name == 'tanh': return tanh
    if name == 'softmax': return softmax
    return linear