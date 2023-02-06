# 04. 数値微分
import numpy as np
from step01 import Variable
from step02 import Square
from step03 import Exp


def numerical_diff(f, x, eps=1e-4):
    """数値微分を行う関数

    Args:
        f (Function): 関数
        x (Variable): 変数
        eps (float, optional): 微小量

    Returns:
        float: 数値微分の結果
    """
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


if __name__ == "__main__":
    f = Square()
    x = Variable(np.array(2.0))
    dy = numerical_diff(f, x)
    print(dy)

    # 合成関数の微分
    def f(x):
        A = Square()
        B = Exp()
        C = Square()
        return C(B(A(x)))

    x = Variable(np.array(0.5))
    dy = numerical_diff(f, x)
    print(dy)
