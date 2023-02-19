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
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))

        self.data = data
        self.grad = None  # 微分値を保持する変数
        self.creator = None  # どの関数によって生成されたかを保持する変数
        self.generation = 0  # どの世代の変数かを保持する変数

    def set_creator(self, func):
        """どの関数によって生成されたかを保持する

        Args:
            func (Function): どの関数によって生成されたか
        """
        self.creator = func
        self.generation = func.generation + 1

    def backward(self):
        """微分を計算する"""
        if self.grad is None:  # 逆伝播の初期値を設定
            self.grad = np.ones_like(self.data)

        funcs = []  # 処理すべき関数の候補
        seen_set = set()

        def add_func(f):
            """関数をfuncsに追加する

            Args:
                f (Function): 追加する関数
            """
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()  # 関数(リストの末尾にある)を取得
            gys = [output.grad for output in f.outputs]  # 出力に対する微分をリストにまとめる
            gxs = f.backward(*gys)  # fの逆伝播
            if not isinstance(gxs, tuple):  # tupleでない場合はtupleに変換
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):  # gxsとf.inputsは各要素が対応しているのでぺアで処理する
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

    def cleargrad(self):
        self.grad = None


def as_array(x):
    """ndarrayに変換する関数

    Args:
        x (ndarray or scalar): 変換する値

    Returns:
        ndarray: 変換された値
    """
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    """関数を表すクラス

    Attributes:
        inputs (list): 入力
        outputs (list): 出力
    """

    def __call__(self, *inputs):  # *により可変長引数を受け取れる
        """関数を呼び出したときの処理

        Args:
            inputs (list): 入力

        Returns:
            list: 出力
        """
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # *によりリストをアンパックして渡す
        if not isinstance(ys, tuple):  # tupleでない場合はtupleに変換
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([input.generation for input in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]  # リストの要素が1つのときは最初の要素だけを返す

    def forward(self, xs):
        """具体的な計算を行う

        Args:
            xs (list): 入力

        Returns:
            list: 関数を適用した出力

        Raises:
            NotImplementedError: 実装されていない場合
        """
        raise NotImplementedError()

    def backward(self, gys):
        """微分を計算する

        Args:
            gys (list): 出力に対する微分値

        Returns:
            list: 入力に対する微分値

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
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    """2乗を計算する関数

    Args:
        x (Variable): 入力

    Returns:
        Variable: 2乗を適用した出力
    """
    return Square()(x)


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
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


def exp(x):
    """指数関数を計算する関数

    Args:
        x (Variable): 入力

    Returns:
        Variable: 指数関数を適用した出力
    """
    return Exp()(x)


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


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)
