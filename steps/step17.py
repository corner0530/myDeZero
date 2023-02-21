# 17. メモリ管理と循環参照
import numpy as np
from step00 import Variable, square

if __name__ == "__main__":
    for i in range(10):
        x = Variable(np.random.randn(10000))
        y = square(square(square(x)))
