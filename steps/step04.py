# 04. 数値微分
import numpy as np
from step00 import Exp, Square, Variable, numerical_diff

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
