# 08. 再帰からループへ
import numpy as np
from step00 import Exp, Square, Variable

if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)  # 逆伝播の初期値は1
    y.backward()
    print(x.grad)
