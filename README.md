# Crash-Resilient Local Coding Agent

Local, multi-process coding-agent engine. Each tenant has its own SQLite database and workspace directory. A fencing token makes a stale worker unable to commit. The worker calls an OpenAI-compatible Chat Completions API with a token you provide.

The build is seven steps, specified in [plan.md](plan.md). Do not start a step until the previous step's pytest gate and `self-test` gate both pass. This guide is the setup and test map. Steps 1 through 7 are implemented: `provision`, `init-task`, `pragma`, `acquire`, `renew`, `release`, `show-lease`, `jail`, `replace`, `run-task`, `demo`, MCP tool calls, `list_files`, `search_text`, `remember`, and `self-test` through step 7.

## Requirements

- Python 3.11 or newer
- The `sqlite3` command-line shell, for inspecting a tenant database by hand

## 1. Local virtualenv

From the repo root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

`.[dev]` installs pytest and pytest-cov. The runtime itself uses the standard library.

## 2. Data root

Tenant files live under a data root. The default is `./var`. Tests and the procedures in `plan.md` pass an explicit root so they do not share state:

```bash
python -m agent_engine --data-root /tmp/agent-demo ...
```

`var/` is gitignored. Each tenant is a directory:

```text
var/tenants/{tenant_id}/engine.db
var/tenants/{tenant_id}/tasks/{task_id}/
```

There is no shared database. `acme` and `globex` are two files.

## 3. Create tenants and a task

```bash
python -m agent_engine --data-root /tmp/agent-demo provision --tenant acme --display-name Acme
python -m agent_engine --data-root /tmp/agent-demo provision --tenant globex --display-name Globex
python -m agent_engine --data-root /tmp/agent-demo init-task --tenant acme --task fix-adder
python -m agent_engine --data-root /tmp/agent-demo pragma --tenant acme
```

`pragma` prints `wal` and `2` (`synchronous=FULL`). The same task id may exist in both tenants. A worker is started with `--tenant` and opens only that tenant's `engine.db`.

Confirm isolation:

```bash
sqlite3 /tmp/agent-demo/tenants/acme/engine.db "SELECT id FROM tasks"
sqlite3 /tmp/agent-demo/tenants/globex/engine.db "SELECT id FROM tasks"
```

## 4. Model credentials

Copy the example and fill in the three values. The file is gitignored.

```bash
cp .env.example .env
```

```bash
export AGENT_LLM_BASE_URL="https://api.openai.com/v1"
export AGENT_LLM_API_KEY="the token you provide"
export AGENT_LLM_MODEL="the model name for that endpoint"
```

Any OpenAI-compatible base URL works. The client POSTs `{AGENT_LLM_BASE_URL}/chat/completions` with `Authorization: Bearer`. The bearer token is redacted in logs and is never written to SQLite.

Two ways to give a tenant a token:

- Shared token. Leave the tenant default in place. Every tenant reads `AGENT_LLM_API_KEY`.
- Per-tenant token. Set that tenant's `llm_api_key_env` to a different variable name, and export that variable. The database stores the variable name, not the secret.

Offline work does not need a token. Pass `--llm canned` to `run-task`, or run `self-test` without `--live`. A live command with no token exits non-zero and names the missing variable.

## 5. Leases

Worker ids are UUIDs, not process ids. `--now` is a unix timestamp used by the framework test so it does not sleep. These commands are implemented. `self-test --step 2` runs the lease procedure. `self-test --through 2` runs steps 1 and 2.

```bash
python -m agent_engine self-test --step 2 --data-root /tmp/agent-step2
python -m agent_engine self-test --through 2 --data-root /tmp/agent-step2
```

```bash
python -m agent_engine --data-root /tmp/agent-demo acquire --tenant acme --task fix-adder --worker-id "$WORKER" --now 0 --ttl 10
python -m agent_engine --data-root /tmp/agent-demo show-lease --tenant acme --task fix-adder
python -m agent_engine --data-root /tmp/agent-demo renew --tenant acme --task fix-adder --worker-id "$WORKER" --token 1 --now 5 --ttl 10
python -m agent_engine --data-root /tmp/agent-demo release --tenant acme --task fix-adder --worker-id "$WORKER" --token 1
```

Renew keeps the same fencing token and sets expiry to `--now` plus `--ttl`. A second worker inside that expiry gets `LeaseNotHeld`. After expiry, acquire returns the next token. An acquire at `--now 0 --ttl 10` followed by renew at `--now 5 --ttl 10` expires at 15, so the next worker acquires at `--now 15`. `globex` is a different file and is unchanged.

## 6. Jail and edits

These commands are implemented. The jail's child environment is `PATH` and `PYTHONPATH` only. `self-test --step 3` runs the timeout and stale-hash procedure. `self-test --through 3` runs steps 1 through 3.

```bash
python -m agent_engine self-test --step 3 --data-root /tmp/agent-step3
python -m agent_engine self-test --through 3 --data-root /tmp/agent-step3
python -m agent_engine jail --timeout 1 -- python fixtures/scripts/spawn_tree.py /tmp/agent-pids
python -m agent_engine replace --root /tmp/agent-step3 --path adder.py --expected-sha256 "$HASH" --old "return a - b" --new "return a + b"
```

The jail starts a new session and, on timeout, kills the process group. That command exits non-zero and prints `ToolTimeout`. Both pids in the file are then dead. A replace whose hash does not match the file exits with `StaleContent` and writes nothing.

## 7. Run a task

`self-test --step 4` is the canned procedure: it prints `completed`, then a cache hit with zero executions and zero posts. `self-test --step 4 --live` calls the real model and exits non-zero when `AGENT_LLM_API_KEY` is unset. `--llm canned` does not need a token. The default `run-task` is the live adapter.

```bash
python -m agent_engine self-test --step 4 --data-root /tmp/agent-step4
python -m agent_engine self-test --through 4 --data-root /tmp/agent-step4
python -m agent_engine self-test --step 4 --live --data-root /tmp/agent-step4-live
```

Canned model, no network:

```bash
python -m agent_engine --data-root /tmp/agent-demo run-task --tenant acme --task fix-adder --llm canned
```

Live model:

```bash
python -m agent_engine --data-root /tmp/agent-demo run-task --tenant acme --task fix-adder-live
```

`completed` is written only after the external verifier passes. A `finish` tool call from the model does not set that status by itself. The verifier runs `checks/commands` in the jail. The Python fixture compiles the work tree and runs `checks/test_adder.py` with `PYTHONPATH` pointed at that tree.

### MCP tools

The model is always offered the built-in tools `read_file`, `replace_text`, `write_file`, `run_checks`, and `finish`. MCP tools are extra. They come from `{data-root}/mcp.json`. With that file absent, nothing is discovered, and a name that starts with `mcp__` is stored as `{"error": "unknown tool"}`.

The repo ships one stdio server, `fixtures/mcp/echo_server.py`. It lists one tool:

| Server | Tool | Name the model calls | Arguments | Result |
| --- | --- | --- | --- | --- |
| `echo` | `ping` | `mcp__echo__ping` | `{"text": "..."}` | JSON text with that `text` and `has_api_key`. The secret value is left out. |

`self-test --step 6` writes the catalog below and runs canned `mcp__echo__ping` on task `mcp-echo`. It prints the echo text. The task stays `running`. The task id is `mcp-echo` so a completed `fix-adder` on the same root is left alone. To offer the same tool to a normal `run-task`, put this next to the data root:

```json
{
  "servers": [
    {
      "name": "echo",
      "argv": ["python", "fixtures/mcp/echo_server.py"],
      "env": []
    }
  ]
}
```

Each entry has `name`, `argv`, and an optional `env` list of variable names. The model cannot add a server or change that argv. A discovered tool is one ReAct step named `mcp__{server}__{tool}`. The engine starts the server with `Jail.run` in the task work root, with a 30 second timeout, once for `tools/list` and again for each `tools/call`. The session is `initialize`, `notifications/initialized`, then that method. The process exits when stdin closes. `AGENT_LLM_API_KEY` is copied only when that entry lists the name. An MCP result does not set `completed` or `rejected`.

### Context budget

`list_files`, `search_text`, and `remember` are ordinary ReAct steps. Search takes a literal substring and stays inside the work root. `remember` stores a path, a quote, and the model's note in that tenant file. The packer runs immediately before each model call. Order is system rules and the task prompt, then knowledge notes newest first, then the newest trajectory steps. `context_token_budget` defaults to 8000. Size is `len(text) // 4`. Set `AGENT_CONTEXT_TOKEN_BUDGET` to change it. Over budget, older tool results become an outcome line and one summary is stored as `context_summary`. The summary does not set `completed` or `rejected`.

`self-test --step 7` runs canned `search_text`, `remember`, and `read_file` on task `context-demo`, then replays that task with a small `context_token_budget`.

### One command for a new project

`agent new PROMPT` is `provision`, `init-task`, `plan`, `accept`, and `apply` in one step. The install puts `agent` on the path, so no `python -m` prefix is needed.

```bash
agent --data-root /tmp/my-project new 'Build a FastAPI service with a health endpoint and pytest tests.'
```

It provisions tenant `acme`, names the task after the prompt, prints the tree, and waits for a `y` on stdin before it writes anything. `--yes` skips that question, `--plan-only` stops after the plan, and `--task` names the task instead of using the slug. A slug that is taken gets a numeric suffix, so the same prompt twice is two tasks. The default check command is this interpreter plus `-m pytest -q`, which avoids a `python` that is not on the jailed `PATH`; `--checks-cmd` overrides it and may be repeated. `--llm canned` runs the whole flow offline. The last lines print the task id, the status, and the work root, plus the `agent apply` line to resume when the loop stopped before `finish`.

### Prompt, plan, then build a tree

The four commands behind `new` are also separate, which is what you want when a human reviews the plan between steps. `run-task` and the demos keep the adder prompt. A new project starts from a prompt instead. `plan` stores an execution plan as `proposed` and writes nothing in the work tree. On an empty task the only tool is `submit_plan`, so that call cannot read or write a file. When `base/` or `work/` already has files, the same call also offers `list_files`, `search_text`, and `read_file`. The model can inspect the tree, and a write tool is rejected, before `submit_plan` is accepted. The printed plan has the approach, the ordered steps, and the file tree. `accept` marks that row `accepted` with no model call and no file write. `apply` refuses a plan that is not accepted, then runs the ReAct loop with the prompt and the accepted plan pinned in the pack. `show-plan` prints the stored prompt, status, and tree.

```bash
python -m agent_engine --data-root /tmp/agent-plan plan --tenant acme --task api-tree --prompt 'Build a small api package with tests.'
python -m agent_engine --data-root /tmp/agent-plan show-plan --tenant acme --task api-tree
python -m agent_engine --data-root /tmp/agent-plan accept --tenant acme --task api-tree
python -m agent_engine --data-root /tmp/agent-plan apply --tenant acme --task api-tree --checks-cmd 'python -m pytest -q'
```

`--llm canned` works on `plan` and `apply` with no token and no network. Re-running `plan` replaces a `proposed` plan and exits non-zero once the plan is `accepted` or the task is terminal. `apply` takes `--max-steps`, default 32, and a tree that hits the cap stays `running` so a later `apply` resumes it. `--checks-cmd` may be repeated and writes `{data-root}/checks/commands` before the loop. Those commands are the operator gate: the model does not choose them and cannot write that directory. Test files the agent writes live inside the task work tree, so pass `--checks-cmd 'python -m pytest -q'` to run them.

`write_file` is a normal ReAct step next to `replace_text`. The path is relative, and `..` or an absolute path returns an error and writes nothing. Parent directories are created with the file, so `services/api/tests/test_app.py` creates the directories on the way down. An existing file is not replaced: updates stay on `replace_text` with the hash from `read_file`. A directory exists because a file was written under it. `apply` creates the work root when the task has no `base/` snapshot, and still seeds `base/` into `work/` when it has one.

## 8. CLI test gates

Unit tests and the CLI self-test are both required.

Offline, steps 1 through 4, no token:

```bash
python -m pytest tests/test_architecture.py tests/test_sqlite_wal.py tests/test_cli_step1.py -q --cov=agent_engine.domain --cov=agent_engine.adapters.sqlite.db --cov=agent_engine.adapters.sqlite.tenant_store --cov-fail-under=90
python -m agent_engine self-test --step 1 --data-root /tmp/agent-step1

python -m pytest tests/test_leases.py tests/test_cli_step2.py -q --cov=agent_engine.application.leases --cov=agent_engine.adapters.sqlite.lease_store --cov-fail-under=90
python -m agent_engine self-test --through 2 --data-root /tmp/agent-step2

python -m pytest tests/test_jail.py tests/test_optimistic_edit.py tests/test_cli_step3.py -q --cov=agent_engine.adapters.exec.jail --cov=agent_engine.adapters.workspace.edits --cov-fail-under=90
python -m agent_engine self-test --through 3 --data-root /tmp/agent-step3

python -m pytest tests/test_architecture.py tests/test_llm_adapter.py tests/test_idempotency.py tests/test_verifier_gate.py tests/test_react_happy.py tests/test_plan_tree.py tests/test_cli_step4.py -q --cov=agent_engine.application --cov=agent_engine.adapters.llm --cov=agent_engine.adapters.verifier --cov=agent_engine.adapters.workspace.plain --cov-fail-under=85 -k "not live"
python -m agent_engine self-test --through 4 --data-root /tmp/agent-step4
```

Live model check for step 4:

```bash
python -m agent_engine self-test --step 4 --live --data-root /tmp/agent-step4-live
```

Step 5 offline (signal steal and copy-on-write rollback), then the live demos:

```bash
python -m pytest tests/test_cow_rollback.py tests/test_demo_crash.py tests/test_demo_suspend.py tests/test_cli_step5.py -q --cov=agent_engine.adapters.workspace.cow --cov=agent_engine.worker --cov=agent_engine.demo --cov-fail-under=80
python -m agent_engine self-test --step 5 --data-root /tmp/agent-step5
python -m agent_engine self-test --step 5 --live --data-root /tmp/agent-step5-live
python -m agent_engine demo --data-root /tmp/agent-step5-live --tenant acme --all
```

Step 6 offline. No token, and no network:

```bash
python -m pytest tests/test_mcp_tools.py tests/test_cli_step6.py -q --cov=agent_engine.adapters.mcp.client --cov-fail-under=85
python -m agent_engine self-test --step 6 --data-root /tmp/agent-step6
python -m agent_engine self-test --through 6 --data-root /tmp/agent-step6
```

Step 7 offline. No token, and no network:

```bash
python -m pytest tests/test_context_budget.py tests/test_cli_step7.py -q --cov=agent_engine.application.context --cov-fail-under=85
python -m agent_engine self-test --step 7 --data-root /tmp/agent-step7
python -m agent_engine self-test --through 7 --data-root /tmp/agent-step7
```

`self-test` prints `PASS` or `FAIL` and exits 0 only when the procedure passes. `--through N` runs the offline procedures for steps 1 through N. `--live` is what turns on the real API.

## 9. Demos

All six run as tenant `acme`. `globex` keeps a sentinel file and an empty task table.

```bash
python -m agent_engine demo --data-root /tmp/agent-step5-live --tenant acme --scenario 1
python -m agent_engine demo --data-root /tmp/agent-step5-live --tenant acme --all
```

| Scenario | What a PASS means |
| --- | --- |
| 1 | The live model repairs the adder and the external verifier passes. |
| 2 | `SIGKILL` after the edit, before the commit. A second worker resumes with the next fence and zero extra writes. |
| 3 | Replaying the committed step is a cache hit: zero side effects and zero extra HTTP posts. |
| 4 | `SIGSTOP`, wait for expiry, the other worker steals, `SIGCONT`. The old token is rejected. |
| 5 | An out-of-band byte is a stale hash. A forced `finish` while tests fail stays `rejected`. |
| 6 | Copy-on-write rollback of the broken edit is under 5 ms, and later step rows are `rolled_back`. |

After `--all`:

```bash
sqlite3 /tmp/agent-step5-live/tenants/globex/engine.db "SELECT COUNT(*) FROM step_intents"
```

That count is `0`.

## Command index

| Goal | Command |
| --- | --- |
| Install for development | `python -m pip install -e ".[dev]"` |
| Create a tenant | `provision --tenant acme --display-name Acme` |
| Create a task | `init-task --tenant acme --task fix-adder` |
| Check WAL mode | `pragma --tenant acme` |
| Take, renew, or drop a lease | `acquire`, `renew`, `release`, `show-lease` |
| Kill a process group on timeout | `jail --timeout 1 -- ...` |
| Compare-and-swap a file | `replace --root ... --expected-sha256 ...` |
| Run offline | `run-task --llm canned` |
| Run against your token | `run-task` with the three env vars set |
| Start a project in one command | `agent new 'Build a ...'` |
| Plan a tree for a prompt | `plan --tenant acme --task api-tree --prompt '...'` |
| Read the stored plan | `show-plan --tenant acme --task api-tree` |
| Accept a plan | `accept --tenant acme --task api-tree` |
| Build the accepted tree | `apply --tenant acme --task api-tree --checks-cmd '...'` |
| CLI gate for one step | `self-test --step N --data-root DIR` |
| CLI gate through step N | `self-test --through N --data-root DIR` |
| Live CLI gate | `self-test --step N --live --data-root DIR` |
| Echo an MCP tool | `self-test --step 6` |
| Search, remember, and summarize | `self-test --step 7` |
| All six demos | `demo --tenant acme --all` |

Prefix tenant commands with `--data-root` when you are not using `./var`. Every command runs as `agent ...` after the editable install, or as `python -m agent_engine ...` without it.
