# Repo Agent Notes

## Commit Shape

- Keep stacked branches linear.
- Each PR branch should contain exactly one commit relative to its base branch.
- If a stack needs to be rewritten, rebuild the branch from its intended base instead of carrying extra parent commits forward.

## Generated Files

- Baseline JSON files are generated artifacts.
- Do not rely on `git commit` hooks to finish staging generated changes for you.
- If a change touches baseline capture, run generation explicitly, then stage the regenerated files before committing.

## Verification

- Local verification must match CI.
- Run `tox -e pre-commit` before committing stack rewrites or generated baseline changes.
- The pre-commit path runs `pre-commit run --all-files`.

## Baseline Capture

- Canonical baseline capture runs on Python 3.11.
- If baseline output changes, regenerate under the canonical interpreter and commit the resulting files.
