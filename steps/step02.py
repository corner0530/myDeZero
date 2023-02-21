# 02. 変数を生み出す関数
import numpy as np
from step00 import Square, Variable

if __name__ == "__main__":
    x = Variable(np.array(10))
    f = Square()
    y = f(x)

    print(type(y))
    print(y.data)
