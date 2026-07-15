# Fix PR

Fix a pull request based on review feedback.

## 1. Identify the PR

```bash
gh pr list
```

## 2. Checkout the PR Branch

```bash
gh pr checkout <PR_NUMBER>
```

## 3. Read Review Comments

```bash
gh pr view <PR_NUMBER> --comments
```

If output is too long, write to a temp file using the Write tool.

Check for conflicts:

```bash
gh pr status --conflict-status
```

If conflict exists, rebase onto the base branch first.

## 4. Plan and Implement Fixes

Switch to plan mode. Address each review comment, then implement.

## 5. Test

Run all relevant tests and verify examples still work.

## 6. Commit and Push

Use the [commit](commit.md) workflow. Push and the PR updates automatically.
