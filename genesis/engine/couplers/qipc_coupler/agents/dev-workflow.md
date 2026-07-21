# QIPCCoupler Development Workflow

## Dependencies

- cuda-graph-qipc (qipc package): build and install into the genesis-world venv
- After modifying cgq, rebuild and reinstall before testing genesis-world

## Iteration Cycle

1. Make changes
2. Run tests
3. If tests fail, fix and re-run
4. Test examples visually with `-v` flag
5. Commit using [commit workflow](commit.md)

## Key Conventions (from CODING_GUIDELINES.md)

- ASCII only in source
- 120-char line wrap
- One option per line for scene-building calls
- Spell names out (no abbreviations)
- Use Genesis domain nouns (entity, link, geom)
- Tests go in `tests/` directory, not inside the coupler module
- Examples use argparse with `-v` flag
- Per-entity config via material fields, not coupler options

## Alignment Testing

The alignment test compares Genesis+QIPCCoupler output against standalone QIPC
on the same URDF. Both must produce identical solver state to machine precision.

## PR Workflow

See [github-pr](github-pr.md) for creating PRs and [fix-pr](fix-pr.md) for
addressing review feedback.
