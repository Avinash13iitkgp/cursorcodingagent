# Crash-Resilient Local Coding Agent

Local, multi-process coding-agent engine. Each tenant has its own SQLite database and workspace directory. A fencing token makes a stale worker unable to commit. The worker calls an OpenAI-compatible Chat Completions API with a token you provide.

The build is six steps, specified in [plan.md](plan.md). Do not start a step until the previous step's pytest gate and `self-test` gate both pass. This guide is the setup and test map. Steps 1 through 4 are implemented: `provision`, `init-task`, `pragma`, `acquire`, `renew`, `release`, `show-lease`, `jail`, `replace`, `run-task`, and `self-test` through step 4. Steps 5 and 6 stay specified in `plan.md`. Step 6 is MCP tool calls.

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

python -m pytest tests/test_architecture.py tests/test_llm_adapter.py tests/test_idempotency.py tests/test_verifier_gate.py tests/test_react_happy.py tests/test_cli_step4.py -q --cov=agent_engine.application --cov=agent_engine.adapters.llm --cov=agent_engine.adapters.verifier --cov=agent_engine.adapters.workspace.plain --cov-fail-under=85 -k "not live"
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
| CLI gate for one step | `self-test --step N --data-root DIR` |
| CLI gate through step N | `self-test --through N --data-root DIR` |
| Live CLI gate | `self-test --step N --live --data-root DIR` |
| All six demos | `demo --tenant acme --all` |

Prefix tenant commands with `--data-root` when you are not using `./var`.
