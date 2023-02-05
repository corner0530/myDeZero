# 01. 箱としての変数

class Variable:
    def __init__(self, data):
        self.data = data

if __name__ == '__main__':
    import numpy as np

    data = np.array(1.0)  # データ
    x = Variable(data)  # 箱
    print(x.data)

    x.data = np.array(2.0)  # 代入
    print(x.data)  # 参照
