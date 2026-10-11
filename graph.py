import numpy as np


class Tensor:
    def __init__(self, op, src0, src1, data, shape: list = None):
        self.data = data  # np.ndarray
        self.src0 = src0
        self.src1 = src1
        self.op = op  # str
        self.shape = shape
        self.grad = None


# ///////////////////////////////////////////////////////////graph


class Graph:
    def __init__(self):
        self.nodes = list()
        self.leafs = list()
        self._seen = set()

    def build_graph_forward_expand(self, root: Tensor):
        assert isinstance(root, Tensor)
        if root in self._seen:
            return
        self._seen.add(root)
        if root.src1 is not None:
            self.build_graph_forward_expand(root.src1)
        if root.src0 is not None:
            self.build_graph_forward_expand(root.src0)
        if (
            root.op is None and root.grad is None
        ):  # grad_w grad_x满足op is None  grad is not None
            self.leafs.append(root)
        else:
            self.nodes.append(root)


def print_graph(g: Graph):
    # print(*(node + " " for node in nodes))
    for node in g.nodes:
        print(node)
    for leaf in g.leafs:
        print(leaf)


# ////////////////////Op
# x * w + b = y
# loss = y_hat - y
# tensor(x) tensor(w) tensor(b) tensor(y)
# =================================================manually build
# matmul = matmul(x, w) add = add(matmul, b)
# sub = sub(y_hat, add)
#   x     w
#  |     |
# +--+--+
#   |        b
# matmul     |
#   +----+--+
#       |
#    add(y)      y_hat
#       |         |
#       +----+---+
#           |
#      sub(loss)
#
# nodes = [matmul,add,sub]
# leafs = [x,w,b,y_hat]
#
# =====================================================auto build
# grad_add = repeat(sub)
# grad_matmul = repeat(grad_add)
# grad_b = repeat(grad_matmul)
# grad_x = mul(grad_matmul, w)
# grad_w = mul(grad_matmul, x)
# only do list append
#
#   x     w
#  |     |              grad_x    grad_w
# +--+--+                |        |
#   |        b          +---+----+
#  matmul    |             |
#   +----+--+         grad_matmul grad_b
#       |                  |        |
#    add(y)      y_hat     +---+---+
#       |         |           |
#       +----+---+       grad_add     grad_y_hat
#           |               |           |
#      sub(loss)------------+----------+
#
# nodes = [matmul,add,sub,grad_add,grad_y_hat,grad_matmul,grad_b,grad_x,grad_w]
# leafs = [x,w,b,y_hat]
# examples:
# Tensor w              op = None, src0 = None, src1 = None, grad = grad_w
# Tensor matmul         op = matmul, src0 = x, src1 = w, grad = grad_matmul
# Tensor sub:           op = sub, src0 = add, src1 = y_hat, grad = 1
# Tensor grad_add:      op = negative, src0 = sub, src1 = None, grad = None
# Tensor grad_y_hat:    op = repeat, src0 = sub, src1 = None, grad = None
# Tensor grad_matmul:   op = repeat, src0 = grad_add, src1 = None, grad = None
# Tensor grad_x:        op = mul, src0 = grad_matmul, src1 = w, grad = None
#
# 梯度节点的 src0 一律指向前向节点(如 grad_matmul = repeat(add)),
# repeat/negative 读的是它的 .grad; mul/matmul 读输入的 .data, 上游梯度以 node.grad 的身份参与


# ////////////////////////////////////////////////ops for build graph


def _new_tensor(
    op: str,
    shape: list,
    src0: Tensor | None = None,
    src1: Tensor | None = None,
):
    return Tensor(op, src0, src1, None, shape)


def add(src0: Tensor, src1: Tensor) -> Tensor:
    return _new_tensor("add", src0.shape, src0, src1)


def sub(src0: Tensor, src1: Tensor) -> Tensor:
    return _new_tensor("sub", src0.shape, src0, src1)


def mul(src0: Tensor, src1: Tensor) -> Tensor:
    return _new_tensor("mul", src0.shape, src0, src1)


def matmul(src0: Tensor, src1: Tensor) -> Tensor:
    return _new_tensor("matmul", [src0.shape[0], src1.shape[1]], src0, src1)


def negative(src0: Tensor) -> Tensor:
    return _new_tensor("negative", src0.shape, src0, None)


def repeat(src0: Tensor) -> Tensor:
    return _new_tensor("repeat", src0.shape, src0, None)


def transpose(src0: Tensor) -> Tensor:
    return _new_tensor("transpose", [src0.shape[1], src0.shape[0]], src0, None)


# //////////////////////////////////////////////build graph

# gf = Graph()
# gf.buld_graph(root)


# def build_graph_expand(g: Graph, node: Tensor):
#     if node in g._seen:
#         return
#     g._seen.add(node)
#     if node.op is None and node.grad is None:
#         g.leafs.append(node)
#     else:
#         g.nodes.append(node)


def _set_grad(gb: Graph, src: Tensor, grad: Tensor):
    # TODO: 改名消除歧义
    src.grad = grad
    gb.build_graph_forward_expand(grad)


def build_graph_backward(gf: Graph) -> Graph:
    gb = Graph()
    gb.nodes = list(gf.nodes)
    gb.leafs = list(gf.leafs)
    gb._seen = set(gf._seen)
    # 倒序遍历: 处理到 node 时, node.grad 已经由它的下游那一步赋好了
    for node in reversed(gf.nodes):
        match node.op:
            case "add":
                _set_grad(gb, node.src0, repeat(node))
                _set_grad(gb, node.src1, repeat(node))
            case "sub":
                _set_grad(gb, node.src0, repeat(node))
                _set_grad(gb, node.src1, negative(node))
            case "mul":
                _set_grad(gb, node.src0, mul(node.grad, node.src1))
                _set_grad(gb, node.src1, mul(node.grad, node.src0))
            case "matmul":  # 注意这里transpose也注册进了计算图
                _set_grad(gb, node.src0, matmul(node.grad, transpose(node.src1)))
                _set_grad(gb, node.src1, matmul(transpose(node.src0), node.grad))
            case _:
                assert False, "op not support"
    return gb


# ///////////////////////////////////////////////////////////compute


def compute_graph(gb: Graph):
    for node in gb.nodes:
        compute(node)


def compute(node: Tensor):
    match node.op:
        case "add":
            compute_add(node)
        case "mul":
            compute_mul(node)
        case "matmul":
            compute_matmul(node)
        case "sub":
            compute_sub(node)
        case "negative":
            compute_negative(node)
        case "repeat":
            compute_repeat(node)
        case "transpose":
            compute_transpose(node)
        case _:
            assert False, "op not support"


def compute_add(node):
    node.data = np.add(node.src0.data, node.src1.data)


def compute_sub(node):
    node.data = np.subtract(node.src0.data, node.src1.data)


def compute_mul(node):
    node.data = np.multiply(node.src0.data, node.src1.data)


def compute_matmul(node):
    node.data = node.src0.data @ node.src1.data


def compute_transpose(node):
    node.data = node.src0.data.T


def _grad_data(g):
    # 种子梯度是 float(main 里 sub.grad = 1.0), 其它梯度是 Tensor, 值在 .data 里
    return g if isinstance(g, (int, float)) else g.data


def compute_negative(node):
    # 反向的 negative: 取上游节点的梯度再取负
    node.data = np.negative(_grad_data(node.src0.grad))


def compute_repeat(node):
    # 反向的 repeat: 把上游梯度广播成自己的形状 (形状相同时就相当于恒等)
    node.data = np.broadcast_to(_grad_data(node.src0.grad), node.shape)


# /////////////////////////////////////////////////Optimizer


def optimize_graph(gb: Graph, lr):
    # 注意 x/y_hat 这种数据叶子也会被一起更新
    for leaf in gb.leafs:
        if leaf.grad is not None:
            leaf.data -= leaf.grad.data * lr
