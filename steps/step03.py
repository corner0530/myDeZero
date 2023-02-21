# 03. 関数の連結
import numpy as np
from step00 import Exp, Square, Variable

if __name__ == "__main__":
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
