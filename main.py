from graph import *

import numpy as np


x = Tensor(None, None, None, np.random.randn(1, 2), [1, 2])
w = Tensor(None, None, None, np.random.randn(2, 1), [2, 1])
b = Tensor(None, None, None, np.random.randn(1, 1), [1, 1])
y_hat = Tensor(None, None, None, np.random.randn(1, 1), [1, 1])

gf = Graph()

matmul = matmul(x, w)
add = add(matmul, b)
sub = sub(y_hat, add)
sub.grad = 1.0

gf.build_graph(sub)

compute_graph(gf)

gb = build_graph_backward(gf)

compute_graph(gb)

lr = 0.01
optimize_graph(gb, lr)
