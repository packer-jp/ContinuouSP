# ContinuouSP
## セットアップ
プロジェクト・パッケージ管理のために [Rye](https://rye-up.com/) を使用しています。
リンク先の指示に従って Rye をインストールしたのち、以下のコマンドを実行することによって `.venv` 下に仮想環境が作られます:

```bash
rye sync
```

コミット時に各種 Formatter/Linter を適用するために、以下のコマンドを仮想環境作成後に一度だけ実行してください:

```bash
rye run pre-commit install
```
