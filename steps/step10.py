# 10. テストを行う
import unittest

import numpy as np
from step00 import Variable, numerical_diff, square


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)  # 期待した値と出力が一致するか検証

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))  # ランダムな入力値を生成
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)  # 2つが近い値かを判定
        self.assertTrue(flg)


if __name__ == "__main__":
    # テストを実行するには以下のいずれか
    # - steps/ に移動し `python -m unittest step10.py`
    # - 以下の関数を呼び出す
    unittest.main()  # テストを実行
