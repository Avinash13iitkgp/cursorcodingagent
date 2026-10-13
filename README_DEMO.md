# LIVE Demo: Plan, Execute, and Test a Project (Agent CLI Only)

Live-model walkthrough using only the `agent` CLI. No `--llm canned`, no `python -m`.
The model plans a real project, builds it with `write_file`, then `run_checks` + `finish`
go through the operator-owned verifier gate.

> Execution guarantee: `apply` runs with the accepted plan pinned in the prompt and
> a reduced toolset — only `read_file`, `replace_text`, `write_file`, `run_checks`,
> `finish` (plus operator MCP tools). `list_files`, `search_text`, and `remember` are
> not offered, so the model executes the accepted steps instead of re-exploring.
> Each step streams to stderr as it happens, followed by a readable pack at the end.

## 0. Prerequisites

- Python 3.11+
- Repo root as cwd
- A live OpenAI-compatible API token

```bash
python3 --version   # >= 3.11
pwd                 # repo root
```

## 1. Setup (once)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
which agent   # should print .venv/bin/agent
```

Always set a scratch root first. **If `$DEMO_ROOT` is empty, the data lands in
`./tenants/` instead of `/tmp/`** (`--data-root ""` resolves to the repo root):

```bash
export DEMO_ROOT=/tmp/my-live-project
rm -rf "$DEMO_ROOT"
echo "DEMO_ROOT=$DEMO_ROOT"
```

> Golden rule: the CLI is stateless — **every** command must pass the **same**
> `--data-root`. Shell vars don't cross terminals, so re-export + `echo` them
> in each new shell. Mixing roots (or an unset var silently resolving to `.`)
> is the #1 cause of `UnknownTenant: acme` (see §7).

## 2. Live model credentials (required)

```bash
export AGENT_LLM_BASE_URL="https://api.openai.com/v1"
export AGENT_LLM_API_KEY="the token you provide"
export AGENT_LLM_MODEL="the model name for that endpoint"
```

Any OpenAI-compatible base URL works. The bearer token is redacted in logs and
never written to SQLite. If any of the three is missing, live commands exit
non-zero and name the missing variable.

Quick check:

```bash
env | grep AGENT_LLM
```

## 3. One-command live demo: plan + execute in a single step

`agent new` = `provision` + `init-task` + `plan` + confirm + `accept` + `apply`.

```bash
agent --data-root "$DEMO_ROOT" new 'Build a small api package with tests.' \
  --yes \
  --checks-cmd 'python -m pytest -q'
```

- `--yes` skips the `accept this plan and build the tree? [y/N]` prompt.
  Without it (on a TTY) you are asked before anything is written.
- `--checks-cmd` is the operator test gate, written to `$DEMO_ROOT/checks/commands`
  **before** the loop. The model cannot edit it. Repeat the flag for multiple commands.
- `--task my-name` overrides the slug-derived task id (`build-a-small-api-package-with-tests`).
  Re-running the same prompt gives `...-2`, `...-3` — nothing is overwritten.

What a live run looks like (real output shape):

```text
# Plan
...
## Steps
1. Create package metadata ...
2. Create error types ...
...
## Files
api/
  __init__.py
  ...
task=build-a-small-api-package-with-tests
status=completed
work=/tmp/my-live-project/tenants/acme/tasks/build-a-small-api-package-with-tests/work
```

`status=completed` means the external verifier re-ran `checks/commands` in the
jail and it passed. `rejected` means the verifier failed. `running` means the
loop stopped before `finish` — resume it (see §6).

Verify the build:

```bash
find "$DEMO_ROOT/tenants/acme/tasks" -type f | sort
cat "$DEMO_ROOT/checks/commands"
cat "$DEMO_ROOT/tenants/acme/tasks/build-a-small-api-package-with-tests/work/api/service.py"
```

## 4. Reviewed live demo: plan → show → accept → apply

Same flow as `new`, split so a human reviews the plan. Uses fixed task id `api-live`.
All five commands must use the **same** `$PLAN_ROOT` — a different root has no
`acme` tenant and fails with `UnknownTenant: acme`.

```bash
export PLAN_ROOT=/tmp/agent-plan-live
rm -rf "$PLAN_ROOT"

# 4a. Tenant + task (empty work tree)
agent --data-root "$PLAN_ROOT" provision --tenant acme --display-name Acme
agent --data-root "$PLAN_ROOT" init-task --tenant acme --task api-live

# 4b. PLAN: calls the live model, stores `proposed`, writes NO files
agent --data-root "$PLAN_ROOT" plan \
  --tenant acme --task api-live \
  --prompt 'Build a small api package with tests.'

# 4c. SHOW: prompt + status + tree
agent --data-root "$PLAN_ROOT" show-plan --tenant acme --task api-live

# 4d. ACCEPT: marks `accepted`. No model call, no file write.
agent --data-root "$PLAN_ROOT" accept --tenant acme --task api-live

# 4e. APPLY: live model builds the tree. Each step streams to stderr as it
# happens (`model: asking...`, `step N: <tool>`, `step N result:`), then a
# readable pack prints at the end (one block per message, not one JSON line).
agent --data-root "$PLAN_ROOT" apply \
  --tenant acme --task api-live \
  --checks-cmd 'python -m pytest -q'
```

The `apply` loop pins your prompt plus the accepted plan into the system rules
(`Execute the accepted plan in order... Do not list, search, or remember.`),
so the model writes the planned files with `write_file` / `replace_text`,
then calls `run_checks` and `finish`. `plan` on a task whose `base/` or `work/`
already has files may offer `list_files` / `search_text` / `read_file` for
inspection — but a write tool during planning is rejected, and `apply` never
offers the explore tools at all.

Plan-only variant of `new` (prints the tree, leaves it `proposed`, writes nothing):

```bash
agent --data-root "$DEMO_ROOT" new 'Build a small api package with tests.' --plan-only
# build it later with:
#   agent --data-root "$DEMO_ROOT" accept --tenant acme --task <slug>
#   agent --data-root "$DEMO_ROOT" apply --tenant acme --task <slug> --checks-cmd 'python -m pytest -q'
```

Rules worth showing:

- `apply` before `accept` → `ValueError: plan ... is proposed, not accepted` (exit 1), writes nothing.
- `plan` after `accept` → `ValueError: plan already accepted` (exit 1). Accepted plans are immutable; use a new `--task` for a new plan.
- Re-`plan` while still `proposed` replaces the row (allowed before `accept`).
- After `plan` but before `apply`, `work/` is empty. After `apply`, the planned files exist:

```bash
find "$PLAN_ROOT/tenants/acme/tasks/api-live" -type f | sort
```

## 5. Test the built project

### 5a. Run the generated tests yourself

```bash
WORK="$PLAN_ROOT/tenants/acme/tasks/api-live/work"
# or for the §3 flow:
# WORK="$DEMO_ROOT/tenants/acme/tasks/build-a-small-api-package-with-tests/work"

cd "$WORK" && PYTHONPATH=. python -m pytest -q
```

Run from the work root with `PYTHONPATH=.` — tests import the package
(e.g. `services.api.app` / `api.service`). Note `list_files` skips cache
directories (`.git`, `.venv`, `__pycache__`, `node_modules`) and `.pyc`/`.tmp`
files, so the model's view of the tree stays clean.

### 5b. The automatic gate (already ran inside `apply`)

The model calls `run_checks` (jailed `checks/commands`, task stays `running`),
then `finish`. `finish` alone changes nothing — the external `VerifyGate`
re-runs the same commands:

- pass → `completed` (terminal)
- fail → `rejected` (terminal)

### 5c. Framework live gate

```bash
agent --data-root /tmp/agent-step4-live self-test --step 4 --live
```

## 6. Resume / reuse

```bash
# Loop stopped before finish (status=running): resume with the same command
agent --data-root "$DEMO_ROOT" apply --tenant acme --task <task-id>

# inspect a plan later
agent --data-root "$DEMO_ROOT" show-plan --tenant acme --task <task-id>

# full six-scenario live suite (tenant must be acme)
agent --data-root /tmp/agent-step5-live demo --tenant acme --all
```

`--max-steps N` caps one run (`run-task` and worker default 8, `apply`/`new`
default 32; all four commands accept the flag).
Hitting the cap leaves the task `running`; the next `apply` resumes it.

## 7. Troubleshooting (live)

| Symptom | Fix |
| --- | --- |
| `missing AGENT_LLM_API_KEY` (or BASE_URL / MODEL) | Export all three (§2). |
| Data lands in `./tenants/` | `$DEMO_ROOT` was empty. `export DEMO_ROOT=/tmp/my-live-project` first. |
| `stdin is not a terminal. pass --yes or --plan-only` | `new` without `--yes` on a non-TTY. Add `--yes` or `--plan-only`. |
| `UnknownTenant: acme` | Wrong `--data-root`: that root has no `acme/engine.db`. This cannot happen inside one `new` (it provisions first) — you ran `accept` / `apply` / `show-plan` / resume under a different root than `provision` / `new`. Diagnose, then rerun under the right root: |
| | `echo "ROOT=[$PLAN_ROOT]"; ls "$PLAN_ROOT/tenants/acme/engine.db"` — the file must exist. If the var is empty, the command silently used the repo root (`.`). |
| `plan ... is proposed, not accepted` | Run `accept` before `apply`. |
| `plan already accepted` | Plans are immutable after accept. New plan = new `--task`. |
| `TaskNotFound: <id>` | You ran `plan` before `init-task`, or typo'd `--task`. `init-task` first. |
| `status=running` after `apply` | Loop stopped before `finish` (cap or model stopped). Resume with `agent apply ...`. |
| `TimeoutError: The read operation timed out` | Live endpoint didn't answer within the 60s socket timeout (this hit on the planning call). Tenant + task were created but **no plan was stored**. Just retry: re-running `new` with the same prompt creates a `<slug>-2` task (harmless), or `plan` manually on the created task id. If it persists, check `AGENT_LLM_BASE_URL` / model health. |
| Pytest import error on generated tests | Run from work root with `PYTHONPATH=.`. |

## 8. Command index (live only)

| Goal | Command |
| --- | --- |
| Install | `python -m pip install -e ".[dev]"` |
| One-command live project | `agent --data-root DIR new 'PROMPT' --yes --checks-cmd 'python -m pytest -q'` |
| Plan only | `agent --data-root DIR new 'PROMPT' --plan-only` |
| Store a plan (live) | `agent --data-root DIR plan --tenant acme --task api-live --prompt '...'` |
| Read the plan | `agent --data-root DIR show-plan --tenant acme --task api-live` |
| Accept the plan | `agent --data-root DIR accept --tenant acme --task api-live` |
| Build the plan (live) | `agent --data-root DIR apply --tenant acme --task api-live --checks-cmd 'python -m pytest -q'` |
| Resume | `agent --data-root DIR apply --tenant acme --task <id>` |
| Test the tree | `cd <work> && PYTHONPATH=. python -m pytest -q` |
| Live framework gate | `agent --data-root DIR self-test --step 4 --live` |
| Live demo suite | `agent --data-root DIR demo --tenant acme --all` |

Full setup + lease/jail/MCP/context reference: `README.md`. Build order: `plan.md`.
