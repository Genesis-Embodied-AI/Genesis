# Commit

Create a well-formatted conventional commit locally. Do NOT push unless the user explicitly asks.

## Steps

### 1. Stage Changes

```bash
git add <files>
```

Review what's staged:

```bash
git diff --cached --stat
```

### 2. Write Commit Message

Write the commit message to `commit_msg.txt` (in repo root, gitignored) using the **Write tool** - NOT echo, cat, or heredoc (PowerShell mangles them).

Follow the [commit convention](commit-convention.md):

```
<type>(<scope>): <short summary>
```

### 3. Commit

```bash
git commit -F commit_msg.txt
rm commit_msg.txt
```

### 4. Push (only if user explicitly requests)

```bash
git push
```

If the branch has no upstream yet:

```bash
git push -u origin HEAD
```
