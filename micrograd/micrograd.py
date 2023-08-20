import math
class Value:

    def __init__(self, data, _children = (), _op = '', label = ''):
        self.data = data
        self.grad = 0.0
        self._prev = _children
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f'Value(data={self.data})'

    def __add__(self, right_operand):
        right_operand = right_operand if isinstance(right_operand, Value) else Value(data = right_operand)
        out = Value(self.data + right_operand.data, (self, right_operand), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            right_operand.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, right_operand):
        out = self + right_operand
        return out

    def __sub__(self, right_operand):
        out = self + (-right_operand)
        return out

    def __neg__(self):
        return -1 * self

    def __mul__(self, right_operand):
        right_operand = right_operand if isinstance(right_operand, Value) else Value(data = right_operand)
        out = Value(self.data * right_operand.data, (self, right_operand), '*')

        def _backward():
            self.grad += right_operand.data * out.grad
            right_operand.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, right_operand):
        out = self * right_operand
        return out
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        out = Value((math.exp(2*x) - 1)/(math.exp(2*x) + 1), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0, (self,), 'ReLU')

        def _backward():
            self.grad += (1 if self.data > 0 else 0) * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
          if v not in visited:
            visited.add(v)
            for child in v._prev:
              build_topo(child)
            topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
          node._backward()