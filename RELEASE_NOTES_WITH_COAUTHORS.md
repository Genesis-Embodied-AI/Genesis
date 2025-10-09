# Release Notes with Co-Authors

This document explains how to generate release notes that include all contributors of merged PRs, including co-authors specified in commit messages.

## Quick Start

The repository includes a release notes generator script that automatically detects and includes co-authors from Git commit trailers.

### Generate Release Notes for Latest Changes

```bash
# Generate release notes since the last tag
python scripts/generate_release_notes.py --since v0.3.3 --version "0.4.0"

# Save to a file
python scripts/generate_release_notes.py --since v0.3.3 --version "0.4.0" --output new_release.md
```

### Example Output

```markdown
## 0.4.0

### Bug Fixes

* Fix point-cloud rendering from Camera depth map. (@ceasor-mao, @duburcqa) (#1512)

### New Features  

* Add support for new rendering backend. (@author1, @coauthor1, @coauthor2) (#1508)

### Miscellaneous

* Improve runtime and compile time performance. (@YilingQiao, @duburcqa) (#1164)
```

## How Co-Authors Are Detected

The script automatically detects co-authors from Git commit message trailers:

```
[MISC] Import math module instead of constants (#1812)

This change helps avoid violating the gstaichi pure checker.

Co-authored-by: Jane Doe <jane@example.com>
Co-authored-by: John Smith <12345+johnsmith@users.noreply.github.com>
```

Result: `(@hughperkins, @jane, @johnsmith)`

## Benefits Over Manual Release Notes

- **Complete Attribution**: Includes all contributors, not just the PR author
- **Automatic Detection**: No manual tracking of co-authors needed  
- **Consistent Format**: Matches existing release note style
- **GitHub Integration**: Converts GitHub emails to usernames automatically
- **No Duplicates**: Removes duplicate contributors within the same PR

## Best Practices for Contributors

### When Making a PR with Multiple Contributors

1. **Use Git Co-authoring**: Add co-author trailers to your commit messages:
   ```
   git commit -m "Add new feature
   
   Co-authored-by: Jane Doe <jane@example.com>
   Co-authored-by: John Smith <john@example.com>"
   ```

2. **GitHub Web Interface**: When merging PRs, GitHub can automatically add co-author trailers if you:
   - Use the "Add co-authors" option in the PR description
   - Or add co-authors in the merge commit message

3. **Pair Programming**: Always credit your pair programming partner:
   ```
   Co-authored-by: Partner Name <partner@example.com>
   ```

### For Maintainers

1. **Encourage Co-authoring**: Remind contributors to add co-authors when applicable
2. **Review Generated Notes**: The script output should be reviewed before publishing
3. **Manual Edits**: Feel free to edit the generated notes for clarity or additional context

## Integration Options

### Manual Process
1. Run the script before each release
2. Copy output to `RELEASE.md`  
3. Review and edit as needed
4. Commit and tag the release

### Automated Process
- Use the included GitHub Actions workflow (`.github/workflows/generate-release-notes.yml`)
- Automatically generates release notes on tag creation
- Creates PRs to update `RELEASE.md`

### Custom Integration  
- The script can be integrated into any CI/CD pipeline
- Supports both tag-based and date-based ranges
- Output can be customized via command-line options

## See Also

- [scripts/README.md](scripts/README.md) - Detailed script documentation
- [Git Co-authoring Guide](https://docs.github.com/en/pull-requests/committing-changes-to-your-project/creating-and-editing-commits/creating-a-commit-with-multiple-authors)
- [GitHub Actions Workflows](.github/workflows/generate-release-notes.yml)