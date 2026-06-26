# Repo Agent Notes

## First Read

- Baseline JSON files under `service_capacity_modeling/tools/data/` are generated fixtures.
- Keep feature, capture, snapshot, and test changes in separate PR layers when possible.
- Reviewers should be able to explain each branch in one sentence.

## Python Style Bias

- Prefer dense, local Python over one-use abstractions. If a helper is only
  called once and mainly names a small loop, a formatted error string, or a
  straightforward model constructor, keep the code inline at the call site.
- Extract a function when it is reused, creates a real planner/model boundary,
  isolates expensive or subtle behavior, or turns a deeply nested block into a
  readable operation. Do not extract only to make code look more layered.
- Prefer the existing accumulator style: initialize `dict` and `list` values
  near the loop, mutate them directly, and return the assembled object at the
  end. Avoid building small private classes unless they hold state across
  multiple methods or are part of a public return contract.
- Prefer explicit loops when they make planner flow obvious. Use
  comprehensions for compact filtering, projection, and `zip`/`merge_plan`
  composition when the surrounding code already does so.
- Keep comments for domain constraints and surprising behavior. Do not add
  comments that only restate the loop or assignment.

Existing style examples to follow narrowly:

- `service_capacity_modeling/capacity_planner.py::_group_plans_by_percentile`
  is a good example of keeping planner orchestration local: for each
  percentile, it plans each sub-model, collects non-empty plans, and merges the
  composed results where the data is assembled. Follow that visible
  loop/collect/merge shape. Do not copy its weaker parts as precedent: avoid
  loose `Any` typing, vague singular names for list values, and implicit empty
  behavior when the new code needs a clearer failure mode.
- `service_capacity_modeling/capacity_planner.py::plan_certain_explained`
  accumulates `all_plans` and `all_excuses` in one pass, then builds the final
  merged plans and family graph without extra one-use wrappers.
- `service_capacity_modeling/models/common.py::cluster_infra_cost` keeps simple
  filtering and cost accumulation local instead of introducing helper
  functions for each branch.

## Internal Tracking

- Do not put Jira issue keys, Jira links, or internal ticket references in
  `AGENTS.md`, `CLAUDE.md`, PR descriptions, or other repo guidance. Track that
  context in Jira and local beads instead.

Avoid patterns like:

- `_format_error_for_one_call_site()` when the f-string is readable inline.
- `_iter_foo()` when the caller immediately converts it to `list()` and the
  loop is only used once.
- `_build_bar()` when the function just passes fields into one model
  constructor and does not hide meaningful domain behavior.

## Stack Shape

- Keep stacked branches linear.
- Each PR branch should contain only the commits for that layer relative to its base branch.
- If a branch picks up extra history, rebuild it from the intended base instead of dragging parent commits forward.
- Common stacked-PR failure mode: a child branch targets the right parent by name, but was branched from the wrong tip or wrong remote copy of that parent.
- When that happens, GitHub shows a misleading stack: PR descriptions look right, but commit ancestry is wrong and child PRs pull in unrelated parent history.
- Before pushing stacked branches, verify both the commit DAG and the PR base branch, especially when `origin` and `upstream` both have similarly named branches.
- If you want stacked PRs, create the stack on the canonical `Netflix-Skunkworks/service-capacity-modeling` repo, not on a personal fork.
- For stacked PRs to work, push the stack base branches to the **canonical repo remote** for `Netflix-Skunkworks/service-capacity-modeling`, not only to your fork.
- That canonical remote is often named `upstream`, but some clones may call it `origin`. Check `git remote -v` and use the remote that points at `Netflix-Skunkworks/service-capacity-modeling`.
- If a child PR bases on a branch that exists only on your fork while GitHub stack logic expects the canonical repo copy, the stack can look right by branch name and still be wrong in practice.

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
