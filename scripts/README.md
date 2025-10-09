# Release Notes Generator with Co-Authors Support

This script generates release notes that include all contributors of merged PRs, including co-authors specified in commit messages using Git trailers.

## Features

- **Co-Author Detection**: Automatically detects and includes co-authors from `Co-authored-by:` trailers in commit messages
- **GitHub Username Extraction**: Converts GitHub noreply emails to usernames (e.g., `@username`)
- **PR Categorization**: Automatically categorizes PRs into:
  - Behavior Changing
  - New Features
  - Bug Fixes
  - Miscellaneous
- **Flexible Date/Tag Ranges**: Support for both git references and date ranges
- **Duplicate Removal**: Ensures each contributor is only listed once per PR

## Usage

### Basic Usage

Generate release notes since a specific tag:
```bash
python scripts/generate_release_notes.py --since v0.3.3 --version "0.4.0"
```

Generate release notes between two references:
```bash
python scripts/generate_release_notes.py --since v0.3.3 --until v0.4.0 --version "0.4.0"
```

### Date-Based Generation

Generate release notes for a specific date range:
```bash
python scripts/generate_release_notes.py --since-date "2024-01-01" --until-date "2024-12-31" --version "0.4.0"
```

### Output to File

Save release notes to a file:
```bash
python scripts/generate_release_notes.py --since v0.3.3 --version "0.4.0" --output RELEASE_NOTES.md
```

## Output Format

The script generates release notes in the same format as the existing `RELEASE.md` file:

```markdown
## 0.4.0

### New Features

* Add new feature X. (@author1, @co-author1, @co-author2) (#1234)
* Implement feature Y. (@author2) (#1235)

### Bug Fixes

* Fix critical bug in module Z. (@author3, @co-author3) (#1236)

### Miscellaneous

* Update documentation. (@author4) (#1237)
```

## Co-Author Detection

The script automatically detects co-authors from Git commit message trailers:

```
Commit Title (#1234)

Some commit description.

Co-authored-by: Jane Doe <jane@example.com>
Co-authored-by: John Smith <12345+johnsmith@users.noreply.github.com>
```

This will result in: `(@original-author, @jane, @johnsmith)`

## Integration

### GitHub Actions

You can integrate this script into your release workflow:

```yaml
name: Generate Release Notes
on:
  release:
    types: [created]

jobs:
  release-notes:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history needed
      
      - name: Generate Release Notes
        run: |
          python scripts/generate_release_notes.py \
            --since ${{ github.event.release.tag_name }} \
            --version ${{ github.event.release.tag_name }} \
            --output release_notes.md
```

### Manual Release Process

1. Tag your release: `git tag v0.4.0`
2. Generate release notes: `python scripts/generate_release_notes.py --since v0.3.3 --version "0.4.0"`
3. Copy the output to your `RELEASE.md` file
4. Review and edit as needed
5. Commit and push

## Requirements

- Python 3.6+
- Git repository with merge commits for PRs
- PR titles should include the PR number in format `(#1234)`

## Limitations

- Only works with merge commits (squash-and-merge won't be detected as PRs)
- Requires PR numbers in commit messages
- Co-authors must be specified using proper Git trailer format
- GitHub username extraction only works with GitHub noreply emails