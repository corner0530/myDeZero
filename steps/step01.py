# 01. 箱としての変数

class Variable:
    """変数を表すクラス

    Attributes:
        data (ndarray): 変数の中身
    """
    def __init__(self, data):
        self.data = data

if __name__ == '__main__':
    import numpy as np

    data = np.array(1.0)  # データ
    x = Variable(data)  # 箱
    print(x.data)

    x.data = np.array(2.0)  # 代入
    print(x.data)  # 参照
