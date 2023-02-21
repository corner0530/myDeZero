# My DeZero

[ゼロから作る Deep Learning ❸](https://www.oreilly.co.jp/books/9784873119069/)を読みながら，[DeZero](https://github.com/oreilly-japan/deep-learning-from-scratch-3)を各ステップごとに実装していく．

[![Unit test](https://github.com/corner0530/myDeZero/actions/workflows/action.yml/badge.svg)](https://github.com/corner0530/myDeZero/actions/workflows/action.yml)

## 環境

- Python 3.9

## DeZero からの変更点

- 各ステップで共通する関数などは`steps/common.py`に集約
- テストに unittest ではなく pytest を使用

## モジュール間での読み込みのための準備

このディレクトリのファイルを読み込むため，最初に以下を実行してインストールする

- conda の場合
  ```bash
  conda develop .
  ```
- pip の場合
  ```bash
  pip install -e .
  ```

アンインストールする場合は

- conda の場合
  ```bash
  conda develop -u .
  ```
- pip の場合
  ```bash
  pip uninstall .
  ```
