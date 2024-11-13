# パッケージのデプロイ

このドキュメントでは、PyPIへのパッケージの自動デプロイ手順について説明します。

## 前提条件

- GitHubアカウント
- PyPIアカウント
- PyPIのAPIトークン

## 設定手順

### 1. PyPI APIトークンの準備

1. PyPIにログインし、アカウント設定からAPIトークンを作成
2. GitHubリポジトリの Settings → Secrets → Actions に移動
3. `PYPI_API_TOKEN` という名前で PyPI のAPIトークンを登録

※2024-11-13現在、 [connect0459](https://github.com/omu-meteorology) が発行したトークンをセットしています。

### 2. パッケージ設定

`pyproject.toml` に以下の設定が必要です：

- プロジェクトのメタデータ（名前、説明、作者など）
- 依存関係
- ビルドシステムの設定
- バージョン管理の設定

重要な設定項目：

```toml
[project]
name = "omu-eddy-covariance"
dynamic = ["version"]  # バージョンは自動設定
description = "渦相関法を主とする大気観測データを解析するパッケージ"
license = "MIT"
requires-python = ">= 3.11"

[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[tool.hatch.version]
source = "vcs"
```

### 3. GitHub Actionsワークフロー

`.github/workflows/deploy.yml` に以下のワークフローを設定します：

```yaml
name: Deploy to PyPI

on:
  release:
    types: [created]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

## デプロイの流れ

1. GitHubでリリースを作成すると、自動的にワークフローが起動
2. コードのチェックアウト
3. Python環境のセットアップ
4. 必要なパッケージのインストール
5. パッケージのビルド
6. PyPIへの公開

## 注意事項

- リリースを作成する前に、すべてのテストが通過していることを確認
- `pyproject.toml` の設定が正しいことを確認
- バージョン番号は Git タグから自動的に設定される
- アップロード後、PyPIでパッケージが正しく公開されていることを確認

## トラブルシューティング

デプロイが失敗した場合は、以下を確認してください：

1. GitHub Secretsに`PYPI_API_TOKEN`が正しく設定されているか
2. パッケージ名がPyPIで利用可能か（重複していないか）
3. ビルドエラーがないか
4. バージョン番号が適切か（既存のバージョンとの重複がないか）
