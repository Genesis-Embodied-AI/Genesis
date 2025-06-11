# Claude Code を使った GitHub PR 作成手順書

このガイドでは、Claude Code を使って簡単に GitHub プルリクエストを作成する方法を説明します。

## 📋 事前準備

### 1. GitHub CLI インストール (初回のみ)
```bash
# macOS
brew install gh

# Windows (Chocolatey)
choco install gh

# Linux (Ubuntu/Debian)
sudo apt install gh
```

### 2. GitHub CLI 認証 (初回のみ)
```bash
gh auth login
```
ブラウザが開くので、GitHubアカウントでログインして認証を完了してください。

## 🚀 Claude Code を使った PR 作成手順

### パターン1: 新規機能追加・修正を行う場合

#### Step 1: Claude Code にコード変更を依頼
```
ユーザー: [具体的な機能追加や修正内容を説明]

例:
- "ログイン機能のバグを修正して"
- "新しいAPIエンドポイントを追加して"
- "テストケースを追加して"
```

#### Step 2: Claude Code に PR 作成を依頼
```
ユーザー: "GitHubにPRを作成してください"
```

Claude Code が自動的に以下を実行します：
- `git status` でファイル変更状況を確認
- `git diff` で変更内容を確認
- `git log` で最近のコミットスタイルを確認
- 適切なコミットメッセージを作成
- ファイルをステージング・コミット
- PR を作成

### パターン2: 既存の変更済みファイルから PR を作成する場合

#### Step 1: 現在の状況を確認
```
ユーザー: "現在の変更をGitHubにPRで送りたいです"
```

#### Step 2: Claude Code に PR 作成を依頼
```
ユーザー: "PRを作成してください"
```

## 🔧 Claude Code が自動実行する処理

### 1. 状況分析
```bash
git status          # 変更ファイル確認
git diff           # 変更内容確認
git log --oneline -5  # コミット履歴確認
```

### 2. コミット作成
```bash
git add [関連ファイル]  # 適切なファイルのみステージング
git commit -m "適切なコミットメッセージ"  # 変更内容に基づいたメッセージ
```

### 3. PR 作成
```bash
# ブランチ作成（必要に応じて）
git checkout -b feature-branch

# リモートにプッシュ
git push origin feature-branch

# PR作成
gh pr create --title "PR タイトル" --body "詳細な説明"
```

## 📝 PR の構成要素

Claude Code が作成する PR には以下が含まれます：

### タイトル
- 変更内容を簡潔に表現
- プロジェクトのタイトル規約に従う

### 説明文
```markdown
## Summary
- 変更の概要（箇条書き）

## Changes
- 具体的な変更内容
- 追加/修正されたファイル

## Test plan
- [x] 実行済みのテスト
- [ ] 今後実行予定のテスト

🤖 Generated with [Claude Code](https://claude.ai/code)
```

## 🔄 よくあるシナリオと対応方法

### シナリオ1: フォークからの PR
```
ユーザー: "フォークしたリポジトリから本家にPRを送りたいです"
```

Claude Code の対応：
1. フォーク先のリモートを確認
2. 適切なリモートにプッシュ
3. 本家リポジトリに対する PR を作成

### シナリオ2: 複数のコミットをまとめて PR
```
ユーザー: "これまでの変更をまとめてPRにしてください"
```

Claude Code の対応：
1. 変更履歴を分析
2. 関連する変更をグループ化
3. 適切なPRタイトルと説明を生成

### シナリオ3: 緊急修正の PR
```
ユーザー: "このバグ修正を緊急でPRにしてください"
```

Claude Code の対応：
1. 迅速なコミット作成
2. 緊急性を示すPRタイトル
3. バグ修正に特化した説明文

## ⚠️ 注意点とベストプラクティス

### DO ✅
- **具体的な指示**: 「○○機能を追加してPRを作成して」
- **変更内容の説明**: どんな機能/修正なのかを明確に
- **PR作成の明示**: 「PRを作成して」と明確に依頼

### DON'T ❌
- **曖昧な指示**: 「何かを修正して」
- **一度に大量の変更**: 複数の機能を一つのPRにまとめる
- **テスト不足**: 動作確認なしでのPR作成

## 🎯 効率的な使い方のコツ

### 1. 段階的な作業
```
# Step 1: 機能実装
ユーザー: "ユーザー認証機能を追加してください"

# Step 2: テスト
ユーザー: "テストケースも追加してください"

# Step 3: PR作成
ユーザー: "これらの変更をPRで送ってください"
```

### 2. 変更内容の事前説明
```
ユーザー: "APIのレスポンス形式を変更しました。この変更をPRにしてください。
変更内容:
- JSON構造を統一
- エラーハンドリング改善
- ドキュメント更新"
```

### 3. PR要件の指定
```
ユーザー: "以下の要件でPRを作成してください:
- レビュワー: @username
- ラベル: enhancement
- マイルストーン: v2.0"
```

## 🔧 トラブルシューティング

### 認証エラーが発生した場合
```bash
gh auth status  # 認証状態確認
gh auth login   # 再認証
```

### リモートリポジトリの問題
```
ユーザー: "リモートリポジトリの設定を確認してください"
```

### PR作成失敗時
```
ユーザー: "PR作成でエラーが出ました。手動で作成する手順を教えてください"
```

## 📚 参考資料

- [GitHub CLI 公式ドキュメント](https://cli.github.com/manual/)
- [GitHub Pull Request ガイド](https://docs.github.com/en/pull-requests)
- [Git ワークフロー](https://docs.github.com/en/get-started/quickstart/github-flow)

---

**💡 Pro Tip**: Claude Code は変更内容を理解してコンテキストに応じた適切なPRを作成します。変更の目的や背景を説明すると、より質の高いPRが作成されます。