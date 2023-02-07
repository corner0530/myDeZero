# 02. 変数を生み出す関数
from common import Square, Variable

if __name__ == "__main__":
    import numpy as np

    x = Variable(np.array(10))
    f = Square()
    y = f(x)

    print(type(y))
    print(y.data)
