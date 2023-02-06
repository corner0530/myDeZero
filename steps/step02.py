# 02. 変数を生み出す関数
from step01 import Variable


class Function:
    """関数を表すクラス

    Args:
        input (Variable): 入力

    Returns:
        Variable: 関数を適用した出力
    """
    def __call__(self, input):  # 引数を与えられるとこのメソッドが呼び出される
        x = input.data  # データを取り出す
        y = self.forward(x)  # 具体的な計算を呼び出す
        output = Variable(y)  # Variableとして返す
        return output

    """具体的な計算を行う

    Args:
        x (ndarray): 入力

    Returns:
        ndarray: 関数を適用した出力

    Raises:
        NotImplementedError: 実装されていない場合
    """
    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    """2乗を計算する関数を表すクラス

    Args:
        input (Variable): 入力

    Returns:
        Variable: 2乗を適用した出力
    """
    def forward(self, x):
        return x ** 2  # 具体的な計算を実装するだけ


if __name__ == "__main__":
    import numpy as np

    x = Variable(np.array(10))
    f = Square()
    y = f(x)

    print(type(y))
    print(y.data)
