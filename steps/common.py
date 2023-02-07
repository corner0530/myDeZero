import numpy as np

class Variable:
    """変数を表すクラス

    Attributes:
        data (ndarray): 変数の中身
    """

    def __init__(self, data):
        self.data = data


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

    def forward(self, x):
        """具体的な計算を行う

        Args:
            x (ndarray): 入力

        Returns:
            ndarray: 関数を適用した出力

        Raises:
            NotImplementedError: 実装されていない場合
        """
        raise NotImplementedError()


class Square(Function):
    """2乗を計算する関数を表すクラス

    Args:
        input (Variable): 入力

    Returns:
        Variable: 2乗を適用した出力
    """

    def forward(self, x):
        return x**2  # 具体的な計算を実装するだけ


class Exp(Function):
    """指数関数を計算する関数

    Args:
        input (Variable): 入力

    Returns:
        Variable: 指数関数を適用した出力
    """
    def forward(self, x):
        return np.exp(x)


def numerical_diff(f, x, eps=1e-4):
    """数値微分を行う関数

    Args:
        f (Function): 関数
        x (Variable): 変数
        eps (float, optional): 微小量

    Returns:
        float: 数値微分の結果
    """
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)
