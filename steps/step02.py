from step01 import Variable


class Function:
    def __call__(self, input):  # 引数を与えられるとこのメソッドが呼び出される
        x = input.data  # データを取り出す
        y = self.forward(x)  # 具体的な計算を呼び出す
        output = Variable(y)  # Variableとして返す
        return output

    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2  # 具体的な計算を実装するだけ


if __name__ == "__main__":
    import numpy as np

    x = Variable(np.array(10))
    f = Square()
    y = f(x)

    print(type(y))
    print(y.data)
