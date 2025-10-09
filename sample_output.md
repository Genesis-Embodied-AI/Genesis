## 0.4.0

*This is a demo showing how the release notes generator handles co-authors.*

### Behavior Changing

* [BREAKING] Change default behavior for collision detection. (@maintainer, Jane Doe, @external-contributor) (#1804)

### New Features

* [FEAT] Add support for new rendering backend with GPU acceleration. (@author1, @coauthor1, @coauthor2) (#1808)

### Bug Fixes

* [BUG FIX] Fix shadow map not properly rendered for objects far away from floor plane. (@duburcqa) (#1810)

### Miscellaneous

* Improve runtime and compile time performance. (@YilingQiao, @duburcqa) (#1806)
* [MISC] Import math module instead of constants to avoid violating gstaichi pure checker. (@hughperkins) (#1812)

### Co-Author Benefits

Notice how the script:
* Lists all contributors, not just the PR author
* Handles both GitHub usernames (@username) and full names
* Preserves the existing release note format
* Removes duplicate contributors within the same PR
* Automatically extracts co-authors from Git commit trailers

