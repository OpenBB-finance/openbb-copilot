---
name: pr-review
description: Review code changes using the project's PR review checklist. Use when asked to review a PR, review code, check code quality, or validate changes before merge.
allowed-tools: Read, Glob, Grep, Bash
---

# PR Review Skill

When asked to review code or a PR, follow the project's review guidelines.

## Instructions

### LLM Review Protocol (must follow)

Scope & evidence:
- Review ONLY changes in the PR diff (or the files explicitly provided). Do not comment on unrelated code.
- Every finding must include evidence: file path + line (or nearby identifier) + what changed.

Severity & precedence:
- Blocking = correctness, security, broken functionality, missing/insufficient tests for meaningful logic changes.
- Should fix = important maintainability/pattern issues that increase risk.
- Suggestions = style/nits; default here unless it meaningfully impacts correctness/security/perf.

Behavior:
- Prefer minimal fixes; don't request large refactors unless they reduce real risk.
- If something can't be verified from the diff (tests passing, runtime behavior), say so and recommend a specific check.
- Do NOT automatically run tests (unit, integration). Only run tests if the user explicitly requests it when invoking the skill. Review test code quality from the diff, but don't execute them.


1. **Read the review checklist**: Start by reading `REVIEW.md` to understand the review criteria
2. **Identify what to review**:
   - If a PR number is given, use `gh pr diff <number>` to get the changes
   - If files are specified, read those files
   - If nothing specified, assume it's the current branch and compare with `origin/main`
3. **Go through each checklist section** from REVIEW.md:
   - Tests (most important)
   - Code Quality
   - Python Type Hints
   - FastAPI Patterns
   - Service Layer
   - API & Data Handling
   - Security
   - Configuration
   - LLM Integration
   - File Organization
4. **Run ruff on changed files**:
   - Get the list of changed `.py` files from the PR
   - Run `ruff check` to check for linting issues
   - Report any issues as **Blocking** (must be fixed before merge)
   - Example workflow:
     ```bash
     # Get changed files
     gh pr diff <number> --name-only | grep '\.py$' > /tmp/pr_files.txt
     # Run ruff check on changed files
     xargs ruff check < /tmp/pr_files.txt 2>&1
     ```
5. **Run mypy on changed files**:
   - Run `mypy` on the changed Python files
   - Report type errors as **Blocking**
   - Example workflow:
     ```bash
     # Run mypy on changed files
     xargs mypy < /tmp/pr_files.txt 2>&1
     ```
6. **Report findings** organized by severity:
   - **Blocking**: Must fix before merge (missing tests, security issues, type errors, lint errors)
   - **Should fix**: Important but not blocking (code quality, patterns)
   - **Suggestions**: Nice to have (minor improvements, nitpicks)
   - **System Impact**: System level integration considerations

## Key files to reference

- `REVIEW.md` - The full review checklist
- `CLAUDE.md` - Project conventions
- `pyproject.toml` - Ruff and mypy configuration

## Output format

```
## PR Review: [description]

### Ruff Issues (Blocking)
- [ ] `openbb_ada/file.py:42` - E501 Line too long
- [ ] `openbb_ada/file.py:55` - F401 Unused import

### Mypy Issues (Blocking)
- [ ] `openbb_ada/file.py:42` - error: Incompatible return value type

### Blocking Issues
- [ ] Issue 1 (file:line)
- [ ] Issue 2 (file:line)

### Should Fix
- [ ] Issue 1
- [ ] Issue 2

### Suggestions
- Suggestion 1
- Suggestion 2

### What's Good
- Positive observation 1
- Positive observation 2

### System Impact
- Summary of downstream effects
```
