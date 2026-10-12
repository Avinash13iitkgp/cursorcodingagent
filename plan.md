# Crash-Resilient Local Coding Agent

The prototype is a single-machine, multi-process engine. Each tenant has its own SQLite file and workspace directory. Safety is the fencing token inside that file. Liveness is a wall-clock lease TTL. The worker calls a real OpenAI-compatible Chat Completions API with a token, base URL, and model name from the environment.

Build the five steps in order. Do not start step N+1 until step N's unit-test command and framework procedure both pass.

## Rules for every step

Each step below uses the same five parts.

- **Architectural choices.** Name the types, functions, SQL, and files. A later step implements a port from an earlier step; it does not grow a second write path.
- **Isolation.** State what this step keeps apart, and the test that fails if the boundary leaks.
- **Documented code.** Every new public module starts with a docstring that names the boundary it owns, the boundary it must not cross, and the errors it raises. Every public function documents arguments, return value, and raised errors. `src/agent_engine/adapters/sqlite/schema.sql` has a one-line comment above each table. Tests have a one-line docstring naming the invariant. Comments do not restate the code.
- **Unit tests.** Tests live under `tests/` and use `tmp_path` for data. They do not sleep, send signals, or call the network unless the test name says `live`. Coverage uses `pytest-cov` on the modules that step owns.
- **Framework test.** After the unit tests pass, run the numbered procedure against a scratch `--data-root`. That procedure is how the step is accepted.

Shared implementation choices:

- Python 3.11+, stdlib runtime. Dev dependencies are `pytest` and `pytest-cov`. `pyproject.toml` sets `pythonpath = ["src"]` and `addopts` so coverage fails under the floor for the modules under test.
- `domain/` and `application/` import neither `sqlite3`, `subprocess`, `os.killpg`, nor `agent_engine.adapters`. `composition.py` is the wiring root, added in step 4.
- CLI from step 1 is `python -m agent_engine --data-root DIR ...`. Each process opens SQLite itself. A connection is never inherited across a subprocess.
- Credentials: `AGENT_LLM_API_KEY`, `AGENT_LLM_BASE_URL`, `AGENT_LLM_MODEL`. The tenant row stores `llm_api_key_env` (the variable name, default `AGENT_LLM_API_KEY`). The secret is never written to SQLite, logs, or the repo. `.gitignore` ignores `.env` and `var/`. `.env.example` lists empty names.
- Inspecting another tenant's file in a test uses `mode=ro` and `query_only=ON`.
- CLI testing is `python -m agent_engine self-test`. It is separate from pytest. `--step N --data-root DIR` runs that step's framework procedure and prints `PASS` or `FAIL` with exit 0 only on `PASS`. `--through N` runs the offline procedures for steps 1 through N. `--live` is required before any command that calls the model. A missing token with `--live` exits non-zero and names the missing variable. Pytest remains the unit-test gate. A step is accepted only when both gates pass.
- `README.md` is the setup guide. Step 1 creates it. Each later step adds that step's commands, environment, and self-test. The guide covers the local venv, the editable install, a shared token, a per-tenant token variable, an offline canned model, and the live demo.

Lease SQL, used by step 2 and unchanged afterward. Steal bumps the token. Renew does not. Clean release bumps and clears the owner. `SIGKILL` cannot release.

```sql
UPDATE tasks
SET fencing_token = fencing_token + 1, lease_owner = :worker, lease_expires_at = :expiry
WHERE tenant_id = :tenant AND id = :task
  AND (lease_owner IS NULL OR lease_expires_at <= :now);

UPDATE tasks
SET lease_expires_at = :expiry
WHERE tenant_id = :tenant AND id = :task
  AND lease_owner = :worker AND fencing_token = :token AND lease_expires_at > :now;

UPDATE tasks
SET fencing_token = fencing_token + 1, lease_owner = NULL, lease_expires_at = NULL
WHERE tenant_id = :tenant AND id = :task
  AND lease_owner = :worker AND fencing_token = :token;
```

A commit's fence predicate is inside the `UPDATE`, not a `SELECT` before `BEGIN IMMEDIATE`. Zero rows: re-read the task. A different token or owner is `FenceRejected`. The same token with no `intent_logged` row is not a lost lease.

```sql
UPDATE step_intents
SET status = 'committed', result_json = :result
WHERE tenant_id = :tenant AND task_id = :task AND step_index = :step
  AND status = 'intent_logged'
  AND EXISTS (
    SELECT 1 FROM tasks
    WHERE tenant_id = :tenant AND id = :task
      AND fencing_token = :token AND lease_owner = :worker
      AND lease_expires_at > :now
  );
```

## Step 1 — Per-tenant durable log (75 min)

### Architectural choices

- `TenantId` is a frozen value object. Accepted form is `^[a-z0-9][a-z0-9_-]{0,63}$`.
- `provision(data_root, tenant_id, display_name) -> Path` creates `var/tenants/{tenant_id}/` mode `0700` and `engine.db`.
- `open_tenant_db(data_root, tenant_id) -> Connection` resolves with `Path.relative_to` and opens only `engine.db`. It sets `journal_mode=WAL`, `synchronous=FULL`, `busy_timeout=5000`. There is no shared catalog and no `ATTACH`.
- On open, the one-row `tenant.id` must equal the directory name. A copied file that disagrees raises `TenantMismatch` before task SQL.
- Schema inside each file: `tenant` (one row, `singleton` CHECK plus UNIQUE), `tasks` PK `(tenant_id, id)`, `step_intents` PK `(tenant_id, task_id, step_index)` status `intent_logged|committed|failed|rolled_back`, `idempotency_cache` PK `(tenant_id, idempotency_key)`, `messages` PK `(tenant_id, task_id, seq)`. Stamp triggers abort an insert whose `tenant_id` disagrees with `tenant.id`.
- `workspace_root` is stored relative to the tenant directory.
- CLI: `provision`, `init-task`, `pragma`, and `self-test --step 1`. `pragma` prints `journal_mode` and `synchronous` for one tenant. `self-test --step 1` runs the framework procedure below and prints `PASS` or `FAIL`.

### Isolation

The database file is the tenant boundary. `acme` and `globex` are two files. The same task id `fix-adder` may exist in both. A path outside `data_root/tenants/{tenant_id}/` raises `TenantMismatch` and creates nothing. Domain modules stay free of SQLite.

### Documented code

- `src/agent_engine/domain/models.py`: docstring lists frozen types and says the package performs no I/O.
- `src/agent_engine/domain/ports.py`: each Protocol method documents the tenant argument and the errors it may raise.
- `src/agent_engine/domain/errors.py`: one line per exception stating when it is raised.
- `src/agent_engine/adapters/sqlite/db.py`: docstring states one connection is one tenant file, and lists the three pragmas.
- `src/agent_engine/adapters/sqlite/schema.sql`: comment above `tenant`, `tasks`, `step_intents`, `idempotency_cache`, `messages`, and each stamp trigger.
- `README.md` starts with: what the engine is, Python 3.11 requirement, venv plus `pip install -e ".[dev]"`, `--data-root`, `provision` / `init-task` / `pragma`, and `self-test --step 1`. It states that steps 2 through 5 are specified in `plan.md` and that their commands are added as those steps land.

### Files

`pyproject.toml`, `README.md`, `src/agent_engine/__init__.py`, `src/agent_engine/__main__.py`, `src/agent_engine/domain/models.py`, `src/agent_engine/domain/ports.py`, `src/agent_engine/domain/errors.py`, `src/agent_engine/adapters/sqlite/db.py`, `src/agent_engine/adapters/sqlite/schema.sql`, `src/agent_engine/adapters/sqlite/tenant_store.py`, `tests/test_architecture.py`, `tests/test_sqlite_wal.py`, `tests/test_cli_step1.py`.

### Unit tests

Coverage floor: 90% on `agent_engine.domain` and `agent_engine.adapters.sqlite`.

- `test_tenant_id_rejects_uppercase_dots_and_separators`
- `test_models_are_frozen`
- `test_domain_imports_neither_sqlite_nor_adapters` (AST walk)
- `test_provision_creates_two_distinct_engine_db_files`
- `test_connection_sets_wal_full_and_busy_timeout`
- `test_same_task_id_is_independent_across_tenant_files`
- `test_second_connection_reads_committed_task`
- `test_open_rejects_path_outside_tenant_dir`
- `test_copied_file_with_wrong_tenant_row_raises`
- `test_missing_directory_raises_unknown_tenant`
- `test_stamp_trigger_rolls_back_mismatched_tenant_id`
- `test_self_test_step1_exits_zero_and_prints_pass` invokes the CLI against `tmp_path` and checks both tenant files.

### Framework test

1. `pip install -e ".[dev]"` from the repo root.
2. `python -m agent_engine --data-root /tmp/agent-step1 provision --tenant acme --display-name Acme`
3. Repeat for `globex`.
4. `python -m agent_engine --data-root /tmp/agent-step1 init-task --tenant acme --task fix-adder`
5. Repeat `init-task` for `globex` with the same task id.
6. `python -m agent_engine --data-root /tmp/agent-step1 pragma --tenant acme` prints `wal` and `2` (`synchronous=FULL`).
7. `sqlite3 /tmp/agent-step1/tenants/acme/engine.db "SELECT id FROM tasks"` prints `fix-adder`. The same query on `globex/engine.db` prints `fix-adder`. `ls` shows two `engine.db` files.

### Exit gate

```bash
python -m pytest tests/test_architecture.py tests/test_sqlite_wal.py tests/test_cli_step1.py -q --cov=agent_engine.domain --cov=agent_engine.adapters.sqlite.db --cov=agent_engine.adapters.sqlite.tenant_store --cov-fail-under=90
python -m agent_engine self-test --step 1 --data-root /tmp/agent-step1
```

## Step 2 — Fenced leases inside one tenant file (60 min)

### Architectural choices

- `LeaseService` is constructed with one `TenantId`, a `LeaseStore` already bound to one connection from `open_tenant_db`, and a `Clock`. The service does not import sqlite3. Production clock is `time.time`. Tests pass `ManualClock`.
- Methods: `acquire(task_id, worker_id, now, ttl) -> Lease`, `renew(lease, now, ttl) -> Lease`, `release(lease, now) -> None`, `commit_marker(lease, now) -> None`.
- `worker_id` is a UUID string minted by the process. Tests mint their own UUIDs.
- `acquire` runs the steal statement inside `BEGIN IMMEDIATE` and returns the new token. Zero rows while the lease is unexpired raises `LeaseNotHeld` and leaves the token unchanged. Zero rows because the task is absent raises `TaskNotFound`.
- `renew` extends expiry and returns the same token. Zero rows raises `FenceRejected`.
- `release` bumps the token and clears the owner so the next `acquire` does not wait out the TTL.
- `commit_marker` writes a sentinel column update using the commit-guard predicate. It exists so the fence can be tested before the ReAct loop.
- CLI adds `acquire`, `renew`, `release`, `show-lease`, and `self-test --step 2`. `show-lease` prints `fencing_token`, `lease_owner`, and `lease_expires_at`. `self-test --step 2` runs the lease procedure and prints `PASS` or `FAIL`. `--through 2` runs steps 1 and 2.

### Isolation

Two workers share a tenant only through that tenant's `engine.db`. A `globex` service has no connection to `acme/engine.db`. A stolen token cannot `commit_marker`. A renew cannot bump the fence. Worker ids are not pids.

### Documented code

- `src/agent_engine/application/leases.py`: docstring states steal versus renew versus release, and that the service never opens a second tenant.
- `src/agent_engine/adapters/sqlite/lease_store.py`: each method's docstring names the SQL predicate and the exception for zero rows.
- `Lease` docstring states the token is the capability to commit and the expiry is only a liveness hint.
- `README.md` gains the lease commands, the UUID worker id, and `self-test --step 2` / `--through 2`.

### Files

`src/agent_engine/application/leases.py`, `src/agent_engine/adapters/sqlite/lease_store.py`, `src/agent_engine/self_test.py`, `tests/test_leases.py`, `tests/test_cli_step2.py`. `__main__.py` dispatches acquire, renew, release, and self-test. The fixed tenants, worker ids, and timestamps live in `self_test.py`. Extend the lease section of `README.md`.

### Unit tests

Coverage floor: 90% on `agent_engine.application.leases` and `agent_engine.adapters.sqlite.lease_store`. No sleeps.

- `test_acquire_idle_task_returns_token_1`
- `test_renew_keeps_token_and_extends_expiry`
- `test_second_worker_inside_ttl_raises_lease_not_held`
- `test_second_worker_at_expiry_receives_token_2`
- `test_stale_token_commit_raises_fence_rejected`
- `test_holder_token_commit_succeeds_after_steal_only_for_new_owner`
- `test_release_lets_new_uuid_acquire_without_waiting`
- `test_globex_connection_cannot_change_acme_fence`
- `test_same_task_id_has_independent_tokens_per_file`
- `test_missing_task_raises_task_not_found`
- `test_two_connections_begin_immediate_serialize_token_bump`
- `test_self_test_step2_prints_pass_and_leaves_globex_token_at_zero`

### Framework test

1. Provision `acme` and `globex` under `/tmp/agent-step2` and init `fix-adder` in both.
2. `acquire --tenant acme --task fix-adder --worker-id <uuid-a> --now 0 --ttl 10`. Output token is `1`.
3. `show-lease --tenant acme --task fix-adder` shows owner `<uuid-a>` and token `1`.
4. `renew` with that token and `--now 5 --ttl 10` still prints token `1`.
5. `acquire` with `<uuid-b>` and `--now 5` exits non-zero with `LeaseNotHeld`. `show-lease` still shows token `1`.
6. `acquire` with `<uuid-b>` and `--now 15` prints token `2`. Renew at `--now 5 --ttl 10` set the expiry to 15, so `--now 11` is still inside the lease.
7. `show-lease --tenant globex --task fix-adder` still shows token `0` and a null owner.
8. Re-run the step 2 unit-test command. Both must pass.

### Exit gate

```bash
python -m pytest tests/test_leases.py tests/test_cli_step2.py -q --cov=agent_engine.application.leases --cov=agent_engine.adapters.sqlite.lease_store --cov-fail-under=90
python -m agent_engine self-test --through 2 --data-root /tmp/agent-step2
```

## Step 3 — Process-group jail and optimistic edits (60 min)

### Architectural choices

- `Jail.run(argv, *, cwd, timeout_s, env) -> CompletedProcess`. `Popen(..., start_new_session=True)`. On timeout, `os.killpg(proc.pid, SIGKILL)`, then `wait`, then `ToolTimeout`.
- The child environment is exactly the mapping passed in. Callers pass `PATH` and `PYTHONPATH` only.
- `fixtures/scripts/spawn_tree.py` writes the leader pid and a child pid, then sleeps. It does not call `setsid`.
- `OptimisticEditor.apply(workspace, edit) -> EditResult` with results `applied`, `already_applied`. It hashes the bytes from `Workspace.read`. The engine does not replace `expected_sha256`.
- `old` must match once. Zero matches with a different hash raises `StaleContent` and writes nothing. Several matches raise `AmbiguousEdit` and write nothing. Postimage hash already present returns `already_applied` and does not open the file for write.
- A match writes a same-directory temp file and `os.replace`. Resume deletes leftover `*.tmp` in the work root.
- Paths are relative. `Path.resolve` must stay under the work root via `relative_to`. `..` and absolute paths raise before any write.
- CLI adds `jail`, `replace`, and `self-test --step 3`. `jail` takes `--timeout` and a command. `replace` takes `--root`, `--path`, `--expected-sha256`, `--old`, and `--new`. `self-test --step 3` runs the jail and stale-hash procedure. `--through 3` stays offline.

### Isolation

The jail's process group is not the worker's process group. Tool paths cannot leave the work root. An edit under `tenants/acme/tasks/...` leaves `tenants/globex/engine.db` and `tenants/globex/tasks/` byte-identical. This step's workspace port is the flat directory later swapped for copy-on-write; the editor depends on the port, not on a raw base path.

### Documented code

- `src/agent_engine/adapters/exec/jail.py`: docstring states `start_new_session`, `killpg`, and that the child env is the mapping it was given.
- `src/agent_engine/adapters/workspace/edits.py`: docstring states the preimage hash, the single-match rule, `already_applied`, and the temp-plus-`os.replace` write.
- `spawn_tree.py` module docstring states it must stay in the jail's session so `killpg` reaches the child.
- `README.md` gains `jail`, `replace`, and `self-test --step 3`, including the expected `ToolTimeout` and `StaleContent` results.

### Files

`src/agent_engine/adapters/exec/jail.py`, `src/agent_engine/adapters/workspace/edits.py`, `fixtures/scripts/spawn_tree.py`, `tests/test_jail.py`, `tests/test_optimistic_edit.py`, `tests/test_cli_step3.py`. Update `README.md`.

### Unit tests

Coverage floor: 90% on `agent_engine.adapters.exec.jail` and `agent_engine.adapters.workspace.edits`.

- `test_timeout_kills_leader_and_child`
- `test_successful_command_returns_stdout`
- `test_child_env_excludes_agent_llm_api_key`
- `test_replace_matching_hash_swaps_bytes`
- `test_stale_hash_writes_nothing`
- `test_already_applied_does_not_open_for_write`
- `test_two_matches_raise_ambiguous_edit`
- `test_dotdot_and_absolute_paths_write_nothing`
- `test_acme_edit_leaves_globex_tree_unchanged`
- `test_resume_removes_orphan_tmp_files`
- `test_self_test_step3_rejects_stale_hash_and_kills_process_group`

### Framework test

1. `python -m agent_engine jail --timeout 1 -- python fixtures/scripts/spawn_tree.py /tmp/agent-step3-pids`. The command exits non-zero with `ToolTimeout`. Both pids in the file are dead (`kill -0` fails).
2. `python -m agent_engine jail --timeout 5 -- python -c "print(123)"` prints `123` and exits 0.
3. Write `/tmp/agent-step3/adder.py` containing `return a - b`. Compute its SHA256.
4. `replace` with that hash, `--old "return a - b"`, `--new "return a + b"` exits 0 and the file contains the new line.
5. Append one byte to the file, then `replace` again with the original hash. The command exits non-zero with `StaleContent` and the appended byte is still there.
6. Re-run the step 3 unit-test command.

### Exit gate

```bash
python -m pytest tests/test_jail.py tests/test_optimistic_edit.py tests/test_cli_step3.py -q --cov=agent_engine.adapters.exec.jail --cov=agent_engine.adapters.workspace.edits --cov-fail-under=90
python -m agent_engine self-test --through 3 --data-root /tmp/agent-step3
```

## Step 4 — ReAct loop, live model, idempotency, verifier (90 min)

### Architectural choices

- `src/agent_engine/adapters/workspace/plain.py` implements `Workspace` as a flat directory. The editor and the loop call this port only.
- `OpenAIChatLlm` POSTs `{base_url}/chat/completions` with `Authorization: Bearer`, `tools` (`read_file`, `replace_text`, `finish`), and stdlib `urllib`. Socket timeout is 60s. A test `Transport` records posts and returns canned bodies. The product worker, the demo, and `test_react_happy_live` construct `OpenAIChatLlm` only.
- `ReactLoop.run` reads the latest step first. `committed` returns the cache and does not insert or call the model. `intent_logged` resumes that row and does not call the model. No row calls the model, then T1 inserts exactly one tool call and commits before any file write. Extra tool calls in one message become later steps.
- Idempotency key is SHA256 of canonical JSON `{tenant_id, task_id, step_index, tool, args}` with `sort_keys=True`.
- T2 writes the step result, the cache row, and the `messages` transcript in one fenced transaction.
- `finish` does not change `tasks.status`. `VerifyGate` runs `python -m unittest` on `fixtures/broken_adder/test_adder.py` from `checks/` outside the tool root, with `PYTHONPATH` pointed at the work tree, inside the jail, with env `PATH` and `PYTHONPATH` only. Pass writes `completed`. Fail writes `rejected`. Both are terminal and use the fence predicate. `FenceRejected` leaves the task `running`.
- A work tree whose `test_adder.py` differs from `checks/` fails the gate. The loop stops at 8 steps.
- The bearer token in the log is redacted. The request uses the env var named by the tenant row.
- `src/agent_engine/composition.py` wires stores, plain workspace, jail, verifier, and the LLM. Missing `AGENT_LLM_API_KEY` fails at startup when the command needs a live model.
- CLI adds `run-task` and `self-test --step 4`. `--llm canned` uses the test transport. The default is the live adapter. `self-test --step 4` runs the canned procedure. `self-test --step 4 --live` runs the real model and refuses to start when `AGENT_LLM_API_KEY` is unset.

### Isolation

Cache rows in `globex/engine.db` are not visible to `acme`. The verifier does not receive the transcript and cannot be pointed at an argv chosen by the model. The model cannot write the check file. Secrets do not enter the jail. T1 is durable before the tool runs, so a crash keeps the intent.

### Documented code

- `src/agent_engine/application/react.py`: docstring lists the read-step, T1, execute, T2 order and the one-tool-call rule.
- `src/agent_engine/application/verify_gate.py`: docstring states it is the only writer of `completed` and `rejected`, and that it does not read the transcript.
- `src/agent_engine/adapters/llm/openai_chat.py`: docstring documents the URL, the bearer header, tool-call parsing, and redaction.
- `src/agent_engine/adapters/verifier/external.py`: docstring documents the fixed command, `checks/`, and the scrubbed environment.
- `fixtures/broken_adder/adder.py`: module docstring states `add` is intentionally `a - b`.
- `README.md` gains `.env` setup, a shared `AGENT_LLM_API_KEY`, a per-tenant `llm_api_key_env`, `run-task --llm canned`, `run-task` against a live base URL, and `self-test --step 4` versus `self-test --step 4 --live`.

### Files

`src/agent_engine/application/react.py`, `src/agent_engine/application/verify_gate.py`, `src/agent_engine/adapters/sqlite/step_store.py`, `src/agent_engine/adapters/llm/openai_chat.py`, `src/agent_engine/adapters/workspace/plain.py`, `src/agent_engine/adapters/verifier/external.py`, `src/agent_engine/composition.py`, `.env.example`, `.gitignore`, `fixtures/broken_adder/adder.py`, `fixtures/broken_adder/test_adder.py`, `tests/test_idempotency.py`, `tests/test_llm_adapter.py`, `tests/test_react_happy.py`, `tests/test_verifier_gate.py`, `tests/test_cli_step4.py`. Extend `tests/test_architecture.py` so `application/` is in the import ban. Update `README.md`.

### Unit tests

Coverage floor: 85% on `agent_engine.application.react`, `agent_engine.application.verify_gate`, `agent_engine.adapters.llm.openai_chat`, `agent_engine.adapters.workspace.plain`, and `agent_engine.adapters.verifier.external`. Offline tests use the canned transport. `test_react_happy_live` is excluded from that floor and runs in the framework procedure.

- `test_application_import_ban`
- `test_llm_posts_chat_completions_with_bearer_and_parses_tool_call`
- `test_llm_log_redacts_bearer_token`
- `test_intent_row_is_durable_before_tool_execution`
- `test_resume_of_intent_logged_skips_model`
- `test_committed_step_is_cache_hit_with_zero_executions_and_zero_posts`
- `test_same_key_on_globex_file_is_a_miss`
- `test_replay_of_applied_edit_is_already_applied_and_writes_nothing`
- `test_one_message_with_two_tool_calls_creates_two_steps`
- `test_finish_does_not_set_completed`
- `test_verifier_pass_sets_completed`
- `test_verifier_fail_sets_rejected`
- `test_modified_check_file_fails_the_gate`
- `test_verifier_child_env_has_no_api_key`
- `test_t2_persists_messages`
- `test_self_test_step4_canned_prints_completed_then_cache_hit`
- `test_self_test_live_without_token_exits_nonzero`

### Framework test

1. Copy `broken_adder` into `/tmp/agent-step4/acme-work` and place the unittest only under `/tmp/agent-step4/checks/`. Provision `acme` and `globex`.
2. `run-task --tenant acme --task fix-adder --llm canned` with a canned body sequence of `read_file`, `replace_text` using the hash from the read, then `finish`. The command prints `completed`. `add(2, 3)` logic in the work tree is `a + b`. `globex/engine.db` has no `step_intents`.
3. Run the same command again. The log shows a cache hit, zero tool executions, and zero HTTP posts.
4. `run-task --llm canned` with `tool_choice` forced to `finish` on a fresh broken tree prints `rejected`. `SELECT status FROM tasks` is `rejected`.
5. Export `AGENT_LLM_BASE_URL`, `AGENT_LLM_API_KEY`, and `AGENT_LLM_MODEL`. `run-task --tenant acme --task fix-adder-live` on a fresh broken copy, without `--llm canned`, exits 0 and prints `completed`.
6. Re-run the step 4 unit-test command, which skips `test_react_happy_live`.

### Exit gate

```bash
python -m pytest tests/test_architecture.py tests/test_llm_adapter.py tests/test_idempotency.py tests/test_verifier_gate.py tests/test_react_happy.py tests/test_cli_step4.py -q --cov=agent_engine.application --cov=agent_engine.adapters.llm --cov=agent_engine.adapters.verifier --cov=agent_engine.adapters.workspace.plain --cov-fail-under=85 -k "not live"
python -m agent_engine self-test --through 4 --data-root /tmp/agent-step4
```

Live CLI check, after exporting the three variables:

```bash
python -m agent_engine self-test --step 4 --live --data-root /tmp/agent-step4-live
```

## Step 5 — Copy-on-write rollback and the six demos (90 min)

### Architectural choices

- `src/agent_engine/adapters/workspace/cow.py` implements the same `Workspace` port. Layout: `tenants/{tenant_id}/tasks/{task_id}/{base,work,snapshots/{step}}`. `composition.py` now wires `CowWorkspace` where step 4 wired `plain`.
- Seed hardlinks `base/` into `work/`. The adapter never opens `base/` for write and never `chmod`s or writes an inode with `st_nlink > 1`. Copy-break replaces the work directory entry with a new inode, then writes.
- `snapshot(step)` hardlinks work files and writes `tombstones.json`. `os.link` raising `EXDEV` fails the test. Snapshot creation may fsync. Rollback does not.
- `rollback(step)` relinks work from that snapshot, restores tombstones, and marks later `step_intents` and cache rows `rolled_back` in that tenant database under the fence. The sub-5ms measurement wraps only the relink.
- `remove_file` is on the workspace port for the CoW test. Model tools remain `read_file`, `replace_text`, and `finish`.
- `src/agent_engine/worker.py` is started with `subprocess`, then opens its own tenant database. It acquires a lease, heartbeats on a second connection every `ttl/3`, and runs `ReactLoop`. If the heartbeat matches zero rows, it closes the HTTP socket and does not start T2. `AGENT_CRASH_AFTER` skips renew and raises `SIGKILL` at the named point. `AGENT_MUTATE_AFTER=read_file` appends one byte before the next tool.
- Model-backed leases are 180s. Demo 2 drops to a 2s TTL after the edit is on disk and stops the heartbeat. Demo 4 uses a 1s TTL.
- `src/agent_engine/demo.py` seeds `acme` and `globex`. Scenarios run as `acme`. `globex` gets a sentinel file and an empty task table. CLI: `demo --tenant acme --scenario N` and `--all`. `self-test --step 5` runs the offline copy-on-write and signal scenarios (4 and 6). `self-test --step 5 --live` runs scenarios 1, 2, 3, and 5, then `--all` is the documented full run.

### Isolation

`base/` and the snapshot inodes stay unchanged when `work/` is edited. Rollback cannot leave a cache row that describes bytes the work tree no longer has. `acme` workers never open `globex/engine.db`. A stopped worker's old token cannot commit after a steal. The successor process does not reuse the dead process's SQLite connection.

### Documented code

- `src/agent_engine/adapters/workspace/cow.py`: docstring states hardlink copy-break, the `nlink > 1` rule, tombstones, and that rollback also rewinds step rows.
- `src/agent_engine/worker.py`: docstring states the second connection used for heartbeat, the crash and mutate hooks, and that `completed` is still only written by `VerifyGate`.
- `src/agent_engine/demo.py`: module docstring lists the six scenarios, the tenant they run as, and the `globex` sentinel.
- `README.md` is complete here. It adds demo scenarios 1 through 6, `self-test --step 5`, `self-test --step 5 --live`, `demo --all`, and a short command index of every setup path.

### Files

`src/agent_engine/adapters/workspace/cow.py`, `src/agent_engine/worker.py`, `src/agent_engine/demo.py`, `tests/test_cow_rollback.py`, `tests/test_demo_crash.py`, `tests/test_demo_suspend.py`, `tests/test_cli_step5.py`. Finish `README.md`.

### Unit tests

Coverage floor: 80% on `agent_engine.adapters.workspace.cow`, `agent_engine.worker`, and `agent_engine.demo`. Signal tests are in this floor. Live chat tests are not; the demo command covers them.

- `test_seed_then_write_changes_only_work_inode`
- `test_copy_break_does_not_modify_base_bytes`
- `test_remove_file_tombstones_and_keeps_base`
- `test_rollback_restores_bytes_and_marks_steps_rolled_back`
- `test_rollback_relink_is_under_5ms`
- `test_sigkill_after_edit_before_t2_resumes_with_token_2_and_zero_writes`
- `test_sigstop_then_steal_then_sigcont_rejects_old_token`
- `test_globex_sentinel_unchanged_after_acme_scenario`
- `test_self_test_step5_offline_covers_suspend_and_rollback`
- `test_readme_lists_install_env_self_test_and_demo`

### Framework test

1. Export `AGENT_LLM_BASE_URL`, `AGENT_LLM_API_KEY`, and `AGENT_LLM_MODEL`.
2. `python -m agent_engine demo --data-root /tmp/agent-step5 --tenant acme --scenario 1`. Expect `PASS` and a repaired adder whose checks pass from `checks/`.
3. `--scenario 2`. Expect `PASS`, fence `1 -> 2`, one committed intent, and zero writes on worker B. Wall time stays near the 2s TTL plus the model call, not a 180s wait.
4. `--scenario 3`. Expect `PASS`, `side_effects=0`, and zero extra HTTP posts.
5. `--scenario 4`. Expect `PASS`. The order in the log is stop, expiry, steal, continue, `FenceRejected` on the pre-stop token.
6. `--scenario 5`. Expect `PASS` for the stale hash and for `rejected` on the forced `finish`. The status row is never `completed`.
7. `--scenario 6`. Expect `PASS` and `rollback_ms` under 5. Query the tenant database: steps after the snapshot are `rolled_back`.
8. `--all` exits 0. `sqlite3 /tmp/agent-step5/tenants/globex/engine.db "SELECT COUNT(*) FROM step_intents"` prints `0`. The sentinel hash matches the value printed at seed time.
9. Re-run the step 1 through step 4 exit gates, then the step 5 unit-test command.

### Exit gate

```bash
python -m pytest tests/test_cow_rollback.py tests/test_demo_crash.py tests/test_demo_suspend.py tests/test_cli_step5.py -q --cov=agent_engine.adapters.workspace.cow --cov=agent_engine.worker --cov=agent_engine.demo --cov-fail-under=80
python -m agent_engine self-test --step 5 --data-root /tmp/agent-step5
```

Live CLI check:

```bash
export AGENT_LLM_BASE_URL="https://api.openai.com/v1"
export AGENT_LLM_API_KEY="the token you provide"
export AGENT_LLM_MODEL="the model name for that endpoint"
python -m agent_engine self-test --step 5 --live --data-root /tmp/agent-step5-live
python -m agent_engine demo --tenant acme --all
```

## Demo map

- Demo 1 happy-path repair: live model in step 4, scenario in step 5.
- Demo 2 `SIGKILL` mid-step: fence in step 2, crash resume in step 5.
- Demo 3 cache hit, zero side effects and zero extra HTTP posts: step 4, scenario in step 5.
- Demo 4 `SIGSTOP` and lease steal: clock-injected steal in step 2, real signals in step 5.
- Demo 5 stale hash and forced `finish` rejected by the verifier: edits in step 3, gate in step 4, live scenario in step 5.
- Demo 6 sub-5ms copy-on-write rollback: step 5.
- Tenant file isolation: step 1. Cross-file lease denial: step 2. Path jail: step 3. Cache file partition: step 4. `globex` sentinel: step 5.

## Done

The prototype is done when every step's pytest gate and `self-test` gate still pass, `README.md` documents every setup path, and `python -m agent_engine demo --tenant acme --all` exits 0, with `globex/engine.db` unchanged. Adding a local tenant creates `var/tenants/{id}/engine.db` and the task directories beside it.
