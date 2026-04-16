# Repo Agent Notes

## First Read

- Baseline JSON files under `service_capacity_modeling/tools/data/` are generated fixtures.
- Keep feature, capture, snapshot, and test changes in separate PR layers when possible.
- Reviewers should be able to explain each branch in one sentence.

## Stack Shape

- Keep stacked branches linear.
- Each PR branch should contain only the commits for that layer relative to its base branch.
- If a branch picks up extra history, rebuild it from the intended base instead of dragging parent commits forward.
- Common stacked-PR failure mode: a child branch targets the right parent by name, but was branched from the wrong tip or wrong remote copy of that parent.
- When that happens, GitHub shows a misleading stack: PR descriptions look right, but commit ancestry is wrong and child PRs pull in unrelated parent history.
- Before pushing stacked branches, verify both the commit DAG and the PR base branch, especially when `origin` and `upstream` both have similarly named branches.

## Baseline Capture Order

When planner output grows, prefer this order:

1. Schema or planner output fields
2. Capture and serializer helpers
3. Regenerated baseline JSON snapshots
4. Regression tests
5. Docs or CI glue only if needed

## What Not To Mix

- Do not mix docs with planner feature changes unless the docs explain that exact branch.
- Do not mix CI or hook rewrites with planner behavior unless the branch needs them to pass.
- Do not land regenerated snapshots before the capture code that produces them.
- Do not wave through large baseline diffs without inspecting why they changed.

## Generated Files

- Run baseline generation explicitly, then stage generated files explicitly.
- Do not rely on git hooks to finish staging snapshot changes for you.
- If planner output changes intentionally, expect baseline fixture diffs and review them.

## Verification

- `tox -e pre-commit` for full repo lint parity.
- `tox -e capture-baseline` to regenerate baseline snapshots.
- `tox -e py312 -- <tests>` for focused validation.
- `tox -e install-hooks` to install the custom git hook from `hooks/pre-commit`.

## Rewrite Flow

- Build a clean branch from the intended base.
- Cherry-pick or reapply only the commits for that layer.
- Run generation and tests before committing.
- Keep the final branch reviewable without reading parent diffs.
