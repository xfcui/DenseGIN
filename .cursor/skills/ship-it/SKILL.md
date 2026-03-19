---
name: ship-it
description: "Update test cases to match the current codebase, run the full suite, fix test failures, and prepare a clean commit and push. Source code is only touched to fix genuine bugs. Use when a user asks to run tests, fix issues, and ship/push changes."
---

# Ship It

## Purpose and scope
Use when the user asks to finalize, ship, or push work. The skill updates test cases to agree with the current codebase, ensures adequate coverage, runs and fixes tests, then commits and pushes. Source code is treated as the ground truth and is never modified unless a genuine bug is found.

## Trigger conditions
- The user explicitly asks to run tests.
- The user requests fixing test failures before a commit.
- The user requests shipping or pushing changes.

## Workflow

### Phase 1 — Analyze codebase changes and update tests

1. Run `git diff HEAD -- src/` and `git diff --cached -- src/` to identify all source changes.
2. For each changed source file, read the current API surface: public classes, `__init__` signatures, `__call__` signatures, module-level constants, and exported names.
3. Read every test file under `tst/` and compare against the source APIs found above:
   - Identify stale imports, renamed symbols, removed parameters, and changed signatures.
   - Identify references to old batch-dict keys or deprecated helper functions.
4. Fix every mismatch found in step 3 — update imports, call sites, assertions, and helper factories **in the test files only** so tests agree with the current source code. Do not modify source code to satisfy outdated tests.

### Phase 2 — Ensure sufficient test coverage

5. For each public class and function in the source, verify that at least one test exercises it. Missing coverage means:
   - A class has no `test_*` method that instantiates it.
   - A function has no `test_*` method that calls it.
   - A code path guarded by `if`/`else` or a keyword argument is never triggered by any test.
6. Add concise new `test_*` methods (or extend existing ones) to fill each gap. Place new tests in the most relevant existing test file; do not create new test files unless a new source module has appeared.
7. When writing new tests:
   - Use the same toy-dataset helper (`_make_toy_dataset` / `_build_toy_dataset`) already present in the file.
   - Keep test data minimal (2–4 graphs, pad_to_multiple=4).
   - Assert shapes, dtypes, finiteness, and key invariants (null-graph exclusion, offset correctness, batch contract keys).
   - Do not duplicate logic already tested elsewhere.

### Phase 3 — Run tests and fix bugs

8. Run the full suite:
   ```
   python -m pytest tst/ -q --tb=short 2>&1
   ```
9. If any test fails:
   a. Read the traceback. Determine if the fault is in a test or in source code.
   b. Fix the **test** with a minimal edit. Only fix source code when there is a genuine bug (e.g. crash, wrong computation, broken invariant). Never change source code merely to make a test pass — update the test instead.
   c. Re-run only the failed test file:
      ```
      python -m pytest tst/<file>.py -q --tb=short 2>&1
      ```
   d. Repeat until no failures remain.
10. Run the full suite one final time to confirm green:
    ```
    python -m pytest tst/ -q --tb=short 2>&1
    ```

### Phase 4 — Commit and push

11. Run `git status` and `git diff` to review all staged and unstaged changes.
12. Stage relevant files (`git add`). Never stage files that contain secrets or large binary artifacts.
13. Commit with an intent-first message (see guidance below).
14. Push the current branch to the remote:
    ```
    git push
    ```

## Commit and message guidance
- Focus on why the change was needed, not just what changed.
- Example:
  - `fix(data): prevent node feature offset mismatch`
  - `feat(model): add regression-safe path for small batches`
  - `test: align test suite with updated dataset/model APIs`

## Safety constraints
- Do not revert unrelated changes outside the requested scope.
- Do not discard or force-push unless explicitly requested.
- Do not commit incomplete or partially tested partial fixes.
- When fixing a test failure, prefer the smallest edit **to the test** that makes it correct and passing.
- Only modify source code to fix genuine bugs (crashes, incorrect results, broken invariants) — never to accommodate stale test expectations.
- Do not remove existing passing tests unless their tested behavior has been intentionally removed from the source.
