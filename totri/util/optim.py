"""

num_iter = 100
t = torch.ones((3, 3), requires_grad=True)
optim = TotriOptimizer([t,], MultiplicativeInterpolator(1.e-1, 1.e-6, 0.9), warmup=5)
for i in range(num_iter):
    optim.zero_grad()
    l = (t - 0.5).abs().sum()
    l.backward()
    optim.step()
"""
import torch

class Interpolator():
    pass

class ConstantInterpolator(Interpolator):

    def __init__(self, constant):
        self.constant = constant

    def __call__(self, _):
        return self.constant

class LinearInterpolator(Interpolator):

    def __init__(self, value_0, value_n, n):
        self.value_0 = value_0
        self.value_n = value_n
        self.n = n

    def __call__(self, i):
        weight = max(min(i / (self.n-1), 1), 0)
        return (1 - weight) * self.value_0 + weight * self.value_n

class MultiplicativeInterpolator(Interpolator):

    def __init__(self, value, target, gamma=0.99):
        self.value = value
        self.target = target
        self.gamma = gamma

    def __call__(self, _):
        self.value = self.gamma * self.value + (1-self.gamma) * self.target
        return self.value

class TotriOptimizer():

    def __init__(self, params, lr_interpolator, warmup = 0, optimizer = 'Adam'):
        self.iteration = 0
        self.optim_type = optimizer
        self.lr = None
        if not isinstance(lr_interpolator, Interpolator):
            lr_interpolator = ConstantInterpolator(lr_interpolator)
        self.lr_interpolator = lr_interpolator
        self.set_params(params, warmup)

    def set_params(self, params, warmup = 0):
        self.set_warmup(warmup)
        lr = self.get_next_lr()
        if self.optim_type == 'SGD':
            self.optim = torch.optim.SGD(params, lr)
        elif self.optim_type == 'Adam':
            self.optim = torch.optim.Adam(params, lr)
        else:
            raise ValueError(f'Unsupported Optimizer {self.optim_type}')

    def zero_grad(self):
        self.optim.zero_grad()

    def set_warmup(self, warmup):
        self.warmup_iter = 0
        self.warmup_total = warmup

    def get_next_lr(self):
        self.lr = self.lr_interpolator(self.iteration)
        if self.warmup_total > 0 and self.warmup_iter < self.warmup_total:
            warmup_weight = ((self.warmup_iter + 1) / (self.warmup_total + 1))**2
            self.warmup_iter += 1
            self.lr *= warmup_weight
        self.iteration += 1
        return self.lr

    def step(self):
        # Update
        self.optim.step()
        # Update lr
        lr = self.get_next_lr()
        for pg in self.optim.param_groups:
            pg['lr'] = lr
