import random
from autodiff import Value

def zero_init(shape):
    return [Value(0.0) for _ in range(shape)]

def uniform_init(shape, low, high, seed=None):
    if seed is not None:
        random.seed(seed)
    return [Value(random.uniform(low, high)) for _ in range(shape)]

def normal_init(shape, mean, variance, seed=None):
    if seed is not None:
        random.seed(seed)
    return [Value(random.gauss(mean, variance**0.5)) for _ in range(shape)]

def linear(x):
    return x

def relu(x):
    out = Value(x.data if x.data > 0 else 0.0, (x,), 'ReLU')
    def _backward():
        x.grad += (1.0 if x.data > 0 else 0.0) * out.grad
    out._backward = _backward
    return out

def sigmoid(x):
    return Value(1.0) / (Value(1.0) + (-x).exp())

def tanh(x):
    return (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())

def softmax(logits):
    max_val = max(l.data for l in logits)
    exps = [(l - Value(max_val)).exp() for l in logits]
    sum_exps = sum(exps, Value(0.0))
    return [e / sum_exps for e in exps]

def mse(y_pred, y_true):
    losses = [(yp - Value(yt))**2 for yp, yt in zip(y_pred, y_true)]
    return sum(losses, Value(0.0)) / Value(len(losses))

def binary_crossentropy(y_pred, y_true):
    losses = []
    for yp, yt in zip(y_pred, y_true):
        yt_val = Value(yt)
        loss = -(yt_val * yp.log() + (Value(1.0) - yt_val) * (Value(1.0) - yp).log())
        losses.append(loss)
    return sum(losses, Value(0.0)) / Value(len(losses))

def categorical_crossentropy(y_pred, y_true):
    losses = [-(Value(yt) * yp.log()) for yp, yt in zip(y_pred, y_true)]
    return sum(losses, Value(0.0))