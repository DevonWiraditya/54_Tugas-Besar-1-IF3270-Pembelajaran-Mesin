import numpy as np
#bonus 40%

def unbroadcast(grad, target_shape):
    grad_shape = grad.shape
    if grad_shape == target_shape:
        return grad
    
    ndims_added = len(grad_shape) - len(target_shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)
    
    for i, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[i] > 1:
            grad = grad.sum(axis=i, keepdims=True)
            
    return grad

class Tensor:
    def __init__(self, data, _children=(), _op='', requires_grad=True):
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data, dtype=np.float64) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Tensor(data={self.data.shape}, requires_grad={self.requires_grad})"

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float64)
        for child in self._prev:
            child.zero_grad()

    def backward(self, gradient=None):
        if not self.requires_grad:
            return
            
        if gradient is None:
            gradient = np.ones_like(self.data, dtype=np.float64)
            
        self.grad = np.array(gradient, dtype=np.float64)
        
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        for node in reversed(topo):
            if node._backward is not None:
                node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), '+', 
                     requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad: self.grad += unbroadcast(out.grad, self.data.shape)
            if other.requires_grad: other.grad += unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data - other.data, (self, other), '-', 
                     requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad: self.grad += unbroadcast(out.grad, self.data.shape)
            if other.requires_grad: other.grad += unbroadcast(-out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), '*',
                     requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad: self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad: other.grad += unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data / (other.data + 1e-15), (self, other), '/',
                     requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad: 
                self.grad += unbroadcast((1.0 / (other.data + 1e-15)) * out.grad, self.data.shape)
            if other.requires_grad: 
                self.grad += unbroadcast((-self.data / ((other.data + 1e-15) ** 2)) * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data ** other, (self,), f'**{other}', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data @ other.data, (self, other), '@',
                     requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad: self.grad += out.grad @ other.data.T
            if other.requires_grad: other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __neg__(self): 
        return self * -1
    
    def __radd__(self, other): 
        return self + other
    
    def __rsub__(self, other): 
        return Tensor(other, requires_grad=False) - self
    
    def __rmul__(self, other): 
        return self * other
    
    def __rtruediv__(self, other): 
        return Tensor(other, requires_grad=False) / self

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                if axis is None:
                    self.grad += np.ones_like(self.data) * out.grad
                else:
                    grad_expanded = np.expand_dims(out.grad, axis=axis) if not keepdims else out.grad
                    self.grad += np.broadcast_to(grad_expanded, self.data.shape)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        out_sum = self.sum(axis=axis, keepdims=keepdims)
        return out_sum / n

    def maximum(self, threshold):
        out = Tensor(np.maximum(self.data, threshold), (self,), 'maximum', requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad += (self.data > threshold).astype(float) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp', requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        val = np.where(self.data >= 0, 
                       1 / (1 + np.exp(-np.clip(self.data, -500, 500))), 
                       np.exp(np.clip(self.data, -500, 500)) / (1 + np.exp(np.clip(self.data, -500, 500))))
        out = Tensor(val, (self,), 'sigmoid', requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad += out.data * (1.0 - out.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,), 'tanh', requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad += (1.0 - out.data**2) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-15), (self,), 'log', requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad += (1.0 / (self.data + 1e-15)) * out.grad
        out._backward = _backward
        return out

    def abs(self):
        out = Tensor(np.abs(self.data), (self,), 'abs', requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad += np.sign(self.data) * out.grad
        out._backward = _backward
        return out

    def clip(self, a_min, a_max):
        out = Tensor(np.clip(self.data, a_min, a_max), (self,), 'clip', requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                mask = (self.data >= a_min) & (self.data <= a_max)
                self.grad += mask.astype(float) * out.grad
        out._backward = _backward
        return out

    @property
    def shape(self):
        return self.data.shape
