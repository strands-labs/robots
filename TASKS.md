# 📋 TASKS — strands-labs/robots Migration Tracker

> **Migration**: `cagataycali/strands-gtc-nvidia` → `strands-labs/robots`
> **Generated**: 2026-03-10 by DevDuck autonomous agent
> **Last Updated**: 2026-03-10T03:47Z

---

## 🔴 CRITICAL PATH — Merge Order

### 1. PR #15 — fix: flake8 E203 black compat `[fix/flake8-e203-black-compat]`
- **Status**: ✅ CI PASSING — REVIEW_REQUIRED
- **Size**: +4 / -2 (1 file)
- **Action**: Approve & merge — unblocks remaining PRs

### 2. PR #8 — Universal policy abstraction (18 VLA providers) `[feat/policy-abstraction]`
- **Status**: ⚠️ CHANGES_REQUESTED → all items addressed
- **Size**: +9,529 / -284 (28 files)
- **Reviewer**: @awsarron
- **All fixes pushed**: operator precedence, Florence2 gate, pickle security, _utils tests
- **Action**: Needs re-review from @awsarron

### 3. PR #9 — MuJoCo simulation (32 robots + zenoh mesh) `[feat/mujoco-simulation]`
- **Status**: ✅ CI PASSING — REVIEW_REQUIRED
- **Size**: +19,590 / -1 (100+ files)
- **Deferred** → Issues #16 (zenoh exceptions), #17 (Renderer cache)

### 4. PR #10 — Training pipeline + GPU backends `[feat/training-pipeline]`
- **Status**: ✅ LINT FIXED by DevDuck (commit `90ea947`) — CI re-running
- **Size**: +23,540 / -1 (42 files)
- **Fixes**: `black` formatting for `lerobot_camera.py` + `asset_converter.py`

### 5. PR #11 — 18 Agent tools `[feat/agent-tools]`
- **Status**: REVIEW_REQUIRED (no CI checks yet)
- **Size**: +6,275 / -13 (14 files)
- **Depends on**: PR #8, PR #10

### 6. PR #12 — Comprehensive test suite `[feat/tests]`
- **Status**: ✅ LINT FIXED by DevDuck (commit `8b04ba7`) — CI re-running
- **Size**: +71,124 / -2 (92 files)
- **Fixes**: `black` formatting for `lerobot_camera.py` + `test_e2e_isaac_validation.py`
- **Action**: Merge LAST

### 7. PR #13 — 10 workshop modules + 50 scripts `[feat/samples]`
- **Status**: ✅ CI PASSING — REVIEW_REQUIRED
- **Size**: +33,568 (65 files)
- **Action**: Independent, can merge anytime

### 8. PR #7 — Documentation site (mkdocs-material) `[feat/docs-site]`
- **Status**: REVIEW_REQUIRED
- **Size**: +1,460 / -1 (40 files)
- **Action**: Independent, can merge anytime

---

## 📌 Open Issues — strands-labs/robots

| # | Title | Priority | Linked PR |
|---|-------|----------|-----------|
| #17 | perf: cache MuJoCo Renderer per-frame allocation | Medium | PR #9 |
| #16 | refactor: tighten exception handling in zenoh_mesh.py | Low | PR #9 |
| #6 | Design for `lerobot_teleoperate.py` | Discussion | — |
| #5 | SIL/HIL Integration with Isaac Sim | Feature | — |
| #4 | Explore Expanded E2E Workloads | Feature | — |
| #3 | Clarify Target Edge Devices | Discussion | — |
| #2 | Generic Framework Integration (ROS2) | Feature | — |
| #1 | Add ModalityTransform Methods in DataConfigs | Feature | — |

## 📌 Open Issues — cagataycali/strands-gtc-nvidia (source)

| # | Title | Status |
|---|-------|--------|
| #250 | Fix security related concerns | Active |
| #246 | Work on reviews | Active |
| #243 | New Robots and Policies in LeRobot | Tracking |
| #240 | 🗺️ ROADMAP: Migration to strands-labs/robots | Master tracker |

## 🤖 Open PRs — cagataycali/strands-gtc-nvidia (source)

| # | Title | Author |
|---|-------|--------|
| #244 | feat: LeRobot v0.5.0 integration | github-actions bot |

---

## 🔄 Migration Status

### Migrated (PRs open on strands-labs/robots):
- ✅ Policy abstraction (18 VLA providers) → PR #8
- ✅ MuJoCo simulation (32 robots) → PR #9
- ✅ Training pipeline (GPU backends) → PR #10
- ✅ Agent tools (18 tools) → PR #11
- ✅ Test suite (92 files) → PR #12
- ✅ Workshop samples (10 modules) → PR #13
- ✅ pyproject.toml update → PR #14 (MERGED)
- ✅ Flake8 compat fix → PR #15
- ✅ Documentation site → PR #7

### Still needs migration:
- 🔲 LeRobot v0.5.0 integration (PR #244 on source)
- 🔲 Security fixes (Issue #250 on source)

---

## ✅ Completed Actions (DevDuck Autonomous)

| Time | Action | Result |
|------|--------|--------|
| 2026-03-10 03:40Z | Created TASKS.md with full repo analysis | ✅ |
| 2026-03-10 03:45Z | Fixed lint on PR #10 `feat/training-pipeline` | ✅ `90ea947` |
| 2026-03-10 03:46Z | Fixed lint on PR #12 `feat/tests` | ✅ `8b04ba7` |

---

## 📊 CI Status Summary

| PR | CI | Lint | Tests |
|----|-----|------|-------|
| #15 | ✅ Pass | ✅ | ✅ |
| #13 | ✅ Pass | ✅ | ✅ |
| #9 | ✅ Pass | ✅ | ✅ |
| #10 | 🔄 Re-running | Fixed | Pending |
| #12 | 🔄 Re-running | Fixed | Pending |
| #8 | No checks | — | — |
| #11 | No checks | — | — |
| #7 | — | — | — |
