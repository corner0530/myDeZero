import sys

import numpy as np
import pytest

sys.path.append(".")
from steps.step00 import Variable, numerical_diff, square


def test_forward():
    x = Variable(np.array(2.0))
    y = square(x)
    expected = np.array(4.0)
    assert y.data == expected  # 期待した値と出力が一致するか検証


def test_backward():
    x = Variable(np.array(3.0))
    y = square(x)
    y.backward()
    expected = np.array(6.0)
    assert x.grad == expected


def test_gradient_check():
    x = Variable(np.random.rand(1))  # ランダムな入力値を生成
    y = square(x)
    y.backward()
    num_grad = numerical_diff(square, x)
    flg = np.allclose(x.grad, num_grad)  # 2つが近い値かを判定
    assert flg


if __name__ == "__main__":
    pytest.main()
