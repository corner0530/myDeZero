# 09. 関数をより便利に
import numpy as np
from common import Variable, exp, square

if __name__ == "__main__":
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))

    y.backward()
    print(x.grad)

    x = Variable(np.array(1.0))  # OK
    x = Variable(None)  # OK
    x = Variable(1.0)  # NG
