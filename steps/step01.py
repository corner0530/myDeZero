# 01. 箱としての変数
import numpy as np
from step00 import Variable

if __name__ == "__main__":
    data = np.array(1.0)  # データ
    x = Variable(data)  # 箱
    print(x.data)

    x.data = np.array(2.0)  # 代入
    print(x.data)  # 参照
