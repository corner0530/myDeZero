import numpy as np
from step01 import Variable
from step02 import Function, Square


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


if __name__ == '__main__':
    A = Square()
    B = Exp()
    C = Square()

    # 以下の4つの変数は全てVariableインスタンス
    # そのため複数の関数を連続して適用できる
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    print(y.data)
