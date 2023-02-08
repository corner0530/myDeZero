# coding: utf-8
"""共通の関数やクラスを定義するモジュール"""
import numpy as np


class Variable:
    """変数を表すクラス

    Attributes:
        data (ndarray): 変数の中身
        grad (ndarray): 微分値
        creator (Function): どの関数によって生成されたか
    """

    def __init__(self, data):
        """コンストラクタ

        Args:
            data (ndarray): 変数の中身
        """
        self.data = data
        self.grad = None  # 微分値を保持する変数
        self.creator = None  # どの関数によって生成されたかを保持する変数

    def set_creator(self, func):
        """どの関数によって生成されたかを保持する

        Args:
            func (Function): どの関数によって生成されたか
        """
        self.creator = func

    def backward(self):
        """微分を計算する"""
        funcs = [self.creator]  # 処理すべき関数をここに順に追加する
        while funcs:
            f = funcs.pop()  # 関数(リストの末尾にある)を取得
            x, y = f.input, f.output  # 関数の入出力を取得
            x.grad = f.backward(y.grad)  # backwardメソッドを呼ぶ

            if x.creator is not None:
                funcs.append(x.creator)  # 1つ前の関数をリストに追加


class Function:
    """関数を表すクラス

    Attributes:
        input (Variable): 入力
        output (Variable): 出力
    """

    def __call__(self, input):  # 引数を与えられるとこのメソッドが呼び出される
        """関数を呼び出したときの処理

        Args:
            input (Variable): 入力

        Returns:
            Variable: 関数を適用した出力
        """
        x = input.data  # データを取り出す
        y = self.forward(x)  # 具体的な計算を呼び出す
        output = Variable(y)  # Variableとして返す
        output.set_creator(self)  # 出力変数に生みの親を覚えさせる
        self.input = input  # backwardのために入力を覚えておく
        self.output = output  # 出力も覚える
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

    def backward(self, gy):
        """微分を計算する

        Args:
            gy (ndarray): 出力に対する微分値

        Returns:
            ndarray: 入力に対する微分値

        Raises:
            NotImplementedError: 実装されていない場合
        """
        raise NotImplementedError()


class Square(Function):
    """2乗を計算する関数を表すクラス

    Attributes:
        input (Variable): 入力
    """

    def forward(self, x):
        """2乗を計算する

        Args:
            x (ndarray): 入力

        Returns:
            ndarray: 2乗を適用した出力
        """
        y = x**2
        return y  # 具体的な計算を実装するだけ

    def backward(self, gy):
        """微分を計算する

        Args:
            gy (ndarray): 出力に対する微分値

        Returns:
            ndarray: 入力に対する微分値
        """
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    """指数関数を計算する関数を表すクラス

    Attributes:
        input (Variable): 入力
    """

    def forward(self, x):
        """指数関数を計算する

        Args:
            x (ndarray): 入力

        Returns:
            ndarray: 指数関数を適用した出力
        """
        return np.exp(x)

    def backward(self, gy):
        """微分を計算する

        Args:
            gy (ndarray): 出力に対する微分値

        Returns:
            ndarray: 入力に対する微分値
        """
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


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
