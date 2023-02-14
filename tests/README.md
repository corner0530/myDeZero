- テスト用ファイルを管理する(予定)
  - 作り方は step10 を参照
  - Chainerを正解と見立てて作ると良い
- まとめて実行するには
  ```bash
  $ python -m unittest discover tests
  ```
  なお`discover`は指定したディレクトリを対象に探索し，`test*.py`というパターンのファイル全てを実行する
