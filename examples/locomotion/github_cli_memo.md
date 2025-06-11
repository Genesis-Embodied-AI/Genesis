# GitHub CLI 使用方法メモ

## 初期セットアップ

### 1. インストール
```bash
# macOS (Homebrew)
brew install gh

# その他のインストール方法
# https://github.com/cli/cli#installation
```

### 2. 認証
```bash
# ブラウザ認証
gh auth login

# トークン認証（推奨）
export GH_TOKEN=your_personal_access_token
# または
gh auth login --with-token < token.txt
```

## プルリクエスト作成

### 基本的な作成
```bash
# 現在のブランチからPR作成
gh pr create

# タイトルと本文を指定
gh pr create --title "PR Title" --body "PR Description"

# 本文をファイルから読み込み
gh pr create --title "PR Title" --body-file description.md

# HEREDOCを使用（複数行の説明）
gh pr create --title "PR Title" --body "$(cat <<'EOF'
## Summary
- 変更の概要

## Changes  
- 具体的な変更内容

## Test plan
- テスト計画

🤖 Generated with [Claude Code](https://claude.ai/code)
EOF
)"
```

### オプション
```bash
# レビュワーを指定
gh pr create --reviewer username1,username2

# アサインを指定
gh pr create --assignee username

# ラベルを指定
gh pr create --label bug,enhancement

# ドラフトPRとして作成
gh pr create --draft

# 特定のベースブランチを指定
gh pr create --base main

# 別のリポジトリへのPR
gh pr create --repo owner/repository
```

## プルリクエスト管理

### 一覧表示
```bash
# 自分のPR一覧
gh pr list

# 全てのPR一覧
gh pr list --state all

# 特定の状態のPR
gh pr list --state open
gh pr list --state closed
gh pr list --state merged
```

### 詳細表示
```bash
# PR詳細表示
gh pr view 123

# PRをブラウザで開く
gh pr view 123 --web

# PR差分表示
gh pr diff 123
```

### PR操作
```bash
# PRをマージ
gh pr merge 123

# PRをクローズ
gh pr close 123

# PRを再オープン
gh pr reopen 123

# PRをチェックアウト
gh pr checkout 123

# PRの準備完了（ドラフト解除）
gh pr ready 123
```

## フォーク・リポジトリ操作

### フォーク作成
```bash
# 現在のリポジトリをフォーク
gh repo fork

# 特定のリポジトリをフォーク
gh repo fork owner/repository

# フォークせずにクローン
gh repo fork owner/repository --clone=false
```

### リポジトリ操作
```bash
# リポジトリをブラウザで開く
gh repo view --web

# リポジトリ情報表示
gh repo view owner/repository

# リポジトリクローン
gh repo clone owner/repository
```

## Issues操作

### Issue作成
```bash
# Issue作成
gh issue create --title "Issue Title" --body "Issue Description"

# テンプレートから作成
gh issue create --template bug_report
```

### Issue管理
```bash
# Issue一覧
gh issue list

# Issue詳細表示
gh issue view 456

# Issueクローズ
gh issue close 456
```

## 認証・設定

### 認証状態確認
```bash
# 認証状態確認
gh auth status

# 認証情報表示
gh auth token
```

### 設定
```bash
# デフォルトエディタ設定
gh config set editor vim

# デフォルトプロトコル設定（ssh/https）
gh config set git_protocol ssh

# 設定一覧表示
gh config list
```

## よく使うワークフロー

### 1. フォーク→PR作成
```bash
# 1. リポジトリをフォーク
gh repo fork original/repository

# 2. フォークをクローン
gh repo clone your-username/repository

# 3. ブランチ作成・作業
git checkout -b feature-branch
# ... 作業 ...
git add .
git commit -m "Feature implementation"

# 4. プッシュ
git push origin feature-branch

# 5. PR作成
gh pr create --title "Add new feature" --body "Description"
```

### 2. 既存リポジトリでPR作成
```bash
# 1. ブランチ作成・作業
git checkout -b feature-branch
# ... 作業 ...
git add .
git commit -m "Feature implementation"

# 2. プッシュ
git push origin feature-branch

# 3. PR作成
gh pr create --title "Add new feature" --body "Description"
```

## トラブルシューティング

### 認証エラー
```bash
# 認証し直し
gh auth logout
gh auth login

# トークンが無効な場合
# GitHub設定でPersonal Access Tokenを再生成
```

### 権限エラー
```bash
# リポジトリの権限確認
gh repo view

# フォークが必要な場合
gh repo fork
```

## 便利なエイリアス

```bash
# ~/.gitconfig に追加
[alias]
    pr = "!gh pr create"
    prs = "!gh pr list"
    prv = "!gh pr view"
```

## Pull Request操作（詳細）

### PRをローカルにpull
```bash
# 特定のPRをローカルブランチにチェックアウト
gh pr checkout 123

# PRの変更をローカルに取得（マージせずに確認）
gh pr diff 123

# PRの最新状態を取得
gh pr view 123 --json commits
```

### PRの更新を取得
```bash
# 元のリポジトリ（upstream）から最新を取得
git remote add upstream https://github.com/original-owner/repository.git
git fetch upstream
git merge upstream/main

# または rebase
git rebase upstream/main

# フォークを最新に更新してからPRを更新
git fetch upstream
git checkout main
git merge upstream/main
git push origin main

# feature branchも更新
git checkout feature-branch
git rebase main
git push --force-with-lease origin feature-branch
```

## マージ後の同期

### PRがマージされた後の同期
```bash
# 1. mainブランチに移動
git checkout main

# 2. upstreamから最新を取得
git fetch upstream

# 3. ローカルのmainを更新
git merge upstream/main

# 4. 自分のフォークも更新
git push origin main

# 5. 作業ブランチを削除（任意）
git branch -d feature-branch
git push origin --delete feature-branch
```

### 定期的な同期（簡単版）
```bash
# GitHub CLIを使った簡単な同期
gh repo sync

# 手動での同期
git fetch upstream && git checkout main && git merge upstream/main && git push origin main
```

## 初回セットアップ（フォーク後）

### upstreamリモートの追加
```bash
# フォーク元のリポジトリをupstreamとして追加
git remote add upstream https://github.com/original-owner/repository.git

# リモート確認
git remote -v
# origin    https://github.com/your-username/repository.git (fetch)
# origin    https://github.com/your-username/repository.git (push)
# upstream  https://github.com/original-owner/repository.git (fetch)
# upstream  https://github.com/original-owner/repository.git (push)
```

## 参考リンク

- [GitHub CLI Documentation](https://cli.github.com/manual/)
- [GitHub CLI Repository](https://github.com/cli/cli)
- [Personal Access Token作成](https://github.com/settings/tokens)