# Create PR for QIPCCoupler

## Prohibitions

- **NEVER push directly to `main`**. All changes go through a feature branch + PR.
- **NEVER run `git push --force`** unless the user explicitly requests it.

## Steps

### 1. Prepare branch

Ensure branch is based on the target (either `main` or `feat/ipc_coupler`):

```bash
git fetch upstream
git rebase upstream/feat/ipc_coupler  # or upstream/main
```

Run tests before pushing:

```bash
python -m pytest tests/test_qipc.py -v -n1
```

Check code against Genesis CODING_GUIDELINES:
- ASCII only in source files
- 120-char line wrap
- One option per line in scene-building calls
- No unnecessary comments
- Spell names out (no abbreviations)

Push:

```bash
git push -u origin HEAD
```

### 2. Create PR

**MUST use `--body-file`** to avoid PowerShell escaping issues.

```bash
# 1. Write body to pr_body.md using the Write tool
# 2. Create PR
gh pr create --title "<type>(<scope>): <summary>" --body-file pr_body.md --base feat/ipc_coupler --repo Genesis-Embodied-AI/genesis-world
# 3. Clean up
rm pr_body.md
```

### 3. PR body format

```markdown
## Summary
- <what changed and why, 1-3 bullet points>

## Test plan
- [ ] All relevant tests passing
- [ ] Examples run correctly
- [ ] Code follows CODING_GUIDELINES.md
```

### 4. After creating

- Return the PR URL to the user
- Monitor CI: `gh pr checks <number> --watch`
- If checks fail, fix and push
