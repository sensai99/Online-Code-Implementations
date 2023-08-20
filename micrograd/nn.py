from micrograd import Value
import random

class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

class Neuron(Module):
    def __init__(self, n_inps):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inps)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((w_* x_ for w_, x_ in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]
    
class Layer(Module):
    def __init__(self, n_inps, n_outs):
        self.neurons = [Neuron(n_inps) for _ in range(n_outs)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params

class MLP(Module):
    def __init__(self, n_layers):
        self.layers = [Layer(n_layers[i], n_layers[i + 1]) for i in range(len(n_layers) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for l in self.layers:
            params.extend(l.parameters()) 
        return params
