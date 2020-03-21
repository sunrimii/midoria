import numpy as np

from .layer import Dense, Conv, BatchNorm


class Optimizer:
    def __init__(self, lr):
        self.lr = lr
        
        self.layers: list

class GD(Optimizer):
    def update(self):
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                layer.gamma += self.lr * layer.dLdgamma
                layer.beta += self.lr * layer.dLdbeta
            elif isinstance(layer, (Dense, Conv)):
                layer.W += self.lr * layer.dLdW
                layer.b += self.lr * layer.dLdb

class Adagrad(Optimizer):
    def update(self):
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                layer.gamma += self.lr * layer.dLdgamma
                layer.beta += self.lr * layer.dLdbeta
            elif isinstance(layer, (Dense, Conv)):
                if not hasattr(self, 'dLdW_history'):
                    layer.dLdW_history = 0
                    layer.dLdb_history = 0
                
                layer.dLdW_history += layer.dLdW ** 2
                layer.dLdb_history += layer.dLdb ** 2
                
                new_wlr = self.lr / ((layer.dLdW_history + 1e-15) ** 0.5)
                new_blr = self.lr / ((layer.dLdb_history + 1e-15) ** 0.5)
                
                layer.W += new_wlr * layer.dLdW
                layer.b += new_blr * layer.dLdb
