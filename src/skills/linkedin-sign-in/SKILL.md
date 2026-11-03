---
name: linkedin-sign-in
description: >-
  Opens LinkedIn login in a persistent browser profile and waits for manual sign-in.
  Use when linkedin-job-search Step 0 fails auth or when a user asks to refresh
  LinkedIn session cookies before browser automation.
disable-model-invocation: true
---

# LinkedIn Sign-In

**All project skills and CLI commands:** [.cursor/skills/SKILL.md](../SKILL.md)

This skill is complete once authentication succeeds. It does not trigger job-search.

## Prerequisites

- `poetry run playwright install chromium`
- Run non-headless with persistent profile:
  - `--user-data-dir .browser_profile`
  - `--profile-directory Default`
  - `--no-headless`

## Commands

Use the same profile flags for both commands:

```bash
BM="poetry run python scripts/browser_manager.py \
  --user-data-dir .browser_profile --profile-directory Default --no-headless"
```

1) Optional pre-check:

```bash
$BM check-auth
```

2) If not authenticated, wait for manual sign-in:

```bash
$BM sign-in --timeout-s 120
```

3) Verify auth after sign-in:

```bash
$BM check-auth
```

## Completion Criteria

- `check-auth` returns JSON with `"ok": true` and `"authenticated": true`.
- Stop here. Do not start linkedin-job-search automatically.

## Failure Behavior

- `check-auth` unauthenticated returns an error payload with:
  - `"error": "linkedin_not_authenticated"`
  - `"action": "invoke_skill:linkedin-sign-in"`
- `sign-in` timeout returns:
  - `"error": "linkedin_sign_in_timeout"`
