# 11. 可変長の引数(順伝播編)
import numpy as np
from common import Add, Variable

if __name__ == "__main__":
    xs = [Variable(np.array(2)), Variable(np.array(3))]
    f = Add()
    ys = f(xs)
    y = ys[0]
    print(y.data)
