# Crash & Restart: Kill a Worker Mid-Step, Resume With a Successor (Agent CLI)

Proves the durability rule: **T1 (intent row) is committed before any file write,
so a `SIGKILL` between the edit and T2 loses nothing.** The successor steals the
expired lease (fence bumps), resumes the open `intent_logged` row, and commits it
with zero extra writes and zero extra model calls.

Three ways to test, easiest first. All use the `agent` CLI (plus
`python -m agent_engine.worker` for the hands-on variant).

## 0. How the mechanism works (30-second version)

- `Worker.serve` acquires a lease (fencing token N), heartbeats `renew` every `ttl/3`
  on a second connection, then runs the ReAct loop.
- Every step is two transactions: **T1** inserts the intent row (`intent_logged`)
  **before** the tool executes; **T2** commits the result under the fence predicate
  (`fencing_token = :token AND lease_owner = :worker AND lease_expires_at > :now`).
- `AGENT_CRASH_AFTER=replace_text` makes the worker shorten its lease to
  `AGENT_CRASH_TTL` (default 2s), stop heartbeating, and `SIGKILL` itself
  **after the bytes hit disk, before T2**. The edit survives; the commit doesn't.
- The successor acquires after expiry → token N+1 → finds the `intent_logged` row →
  commits it. The dead worker's token can never commit afterward.

## 1. Push-button live crash test (needs a token)

One command. Live model does the edit; the verification of the resume is local.

```bash
export AGENT_LLM_BASE_URL="https://api.openai.com/v1"
export AGENT_LLM_API_KEY="the token you provide"
export AGENT_LLM_MODEL="the model name for that endpoint"

export CRASH_ROOT=/tmp/agent-crash-live
rm -rf "$CRASH_ROOT"
agent --data-root "$CRASH_ROOT" demo --tenant acme --scenario 2
```

Use a **fresh root every run**: the scenario seeds task `crash-adder`, and a
re-run against a completed tree would not crash at the same point.

What happens, in order:

1. Worker A (live model, `ttl=180`) reads the broken adder and writes the fix.
2. `AGENT_CRASH_AFTER=replace_text` fires: lease shortened to 2s, heartbeat
   stopped, `SIGKILL`. The process dies (negative return code) with `return a + b`
   already on disk but step 0 still `intent_logged`.
3. The demo waits out the ~2s expiry, then starts worker B
   (`AGENT_CANNED=stop`, `AGENT_RESUME_ONLY=1` — no model calls).
4. Worker B steals the lease (token 1 → 2), resumes the open intent, commits it.

Expected tail of output (PASS conditions asserted by the demo itself):

```text
fence 1 -> 2
one committed intent
writes_b=0
token=2
status=running
writes=0
posts=0
executions=1
scenario 2 PASS
```

What each line proves: `token=2` is the steal; `writes=0` (plus an unchanged file
inode, which the demo checks) means the successor did not re-apply the edit;
`posts=0` means no model call was needed; `executions=1` is the single resumed
commit. `status=running` is correct — the crash task is never finished, only resumed.

Inspect the evidence afterward:

```bash
python -c "
import sqlite3
c = sqlite3.connect('file:$CRASH_ROOT/tenants/acme/engine.db?mode=ro', uri=True)
print('TASK:', c.execute(\"SELECT id, status, fencing_token FROM tasks WHERE id='crash-adder'\").fetchall())
print('STEPS:', c.execute(\"SELECT step_index, tool_name, status FROM step_intents WHERE task_id='crash-adder'\").fetchall())
"
grep -n 'return' "$CRASH_ROOT/tenants/acme/tasks/crash-adder/work/adder.py"
```

Expect `fencing_token` 2, one `committed` step, and `return a + b` on disk.

## 2. Hands-on offline crash test (no token, verified step by step)

Drives `python -m agent_engine.worker` directly with a canned model, so you can
inspect the database between the kill and the resume. Every output below was
observed on this machine.

```bash
export ROOT=/tmp/agent-crash-manual
rm -rf "$ROOT"

# 2a. Seed: tenant + task + broken adder in base/ + operator checks, then seed work/
agent --data-root "$ROOT" provision --tenant acme --display-name Acme
agent --data-root "$ROOT" init-task --tenant acme --task crash-adder
mkdir -p "$ROOT/tenants/acme/tasks/crash-adder/base" "$ROOT/checks"
cp fixtures/broken_adder/adder.py "$ROOT/tenants/acme/tasks/crash-adder/base/adder.py"
cp fixtures/broken_adder/test_adder.py "$ROOT/checks/test_adder.py"
python -c "
from pathlib import Path
from agent_engine.adapters.workspace.cow import CowWorkspace
CowWorkspace(Path('$ROOT/tenants/acme/tasks/crash-adder')).seed()
"
```

```bash
# 2b. Worker A: canned single replace, then SIGKILLs itself before T2
AGENT_CRASH_AFTER=replace_text AGENT_CRASH_TTL=2 AGENT_CANNED=replace \
  python -m agent_engine.worker --data-root "$ROOT" --tenant acme --task crash-adder \
  --llm canned --ttl 180
echo "worker A exit=$?"   # 137 = SIGKILL
```

```bash
# 2c. Post-crash state: edit on disk, intent open, fence still 1
grep -n 'return' "$ROOT/tenants/acme/tasks/crash-adder/work/adder.py"  # return a + b
python -c "
import sqlite3
c = sqlite3.connect('file:$ROOT/tenants/acme/engine.db?mode=ro', uri=True)
print('TASK:', c.execute(\"SELECT id, status, fencing_token FROM tasks WHERE id='crash-adder'\").fetchall())
print('STEPS:', c.execute(\"SELECT step_index, tool_name, status FROM step_intents WHERE task_id='crash-adder'\").fetchall())
"
# TASK: [('crash-adder', 'running', 1)]
# STEPS: [(0, 'replace_text', 'intent_logged')]
```

```bash
# 2d. Wait out the 2s crash lease, then resume with a model-free successor
sleep 3
AGENT_CANNED=stop AGENT_RESUME_ONLY=1 \
  python -m agent_engine.worker --data-root "$ROOT" --tenant acme --task crash-adder \
  --llm canned --ttl 30
echo "successor exit=$?"   # 0
```

Expected successor report:

```text
token=2
status=running
writes=0
posts=0
executions=1
```

The steal (`token=2`), the zero-write resume (`writes=0`, `posts=0`), and the
single commit (`executions=1`) — same shape as the live scenario, fully offline.

## 3. Suspend / steal test (offline, no token)

The lease-theft half without a kill: worker A is `SIGSTOP`ped mid-read, worker B
steals after a 1s expiry, worker A is `SIGCONT`ued and its stale token is rejected.

```bash
agent --data-root /tmp/agent-crash-offline demo --tenant acme --scenario 4
```

Expected output:

```text
stop
expiry
steal
continue
FenceRejected
scenario 4 PASS
```

Post-run DB state (verified): `suspend-adder` is `running` with `fencing_token=2`
owned by worker B, and its step 0 is still `intent_logged` — T1 survived the stop,
T2 was correctly refused on the stale token. The old worker prints `FenceRejected`
and commits nothing.

## 4. Offline gate (both crash-adjacent scenarios)

```bash
agent --data-root /tmp/agent-crash-offline self-test --step 5
```

Runs scenario 4 (suspend/steal) and scenario 6 (copy-on-write rollback,
`rollback_ms` under 5ms) with the canned model. Verified: `PASS` in ~1.5s.

```text
stop
expiry
steal
continue
FenceRejected
scenario 4 PASS
rollback_ms=2.368
scenario 6 PASS
PASS
```

Live gate (scenarios 1, 2, 3, 5 against the real model):

```bash
agent --data-root /tmp/agent-step5-live self-test --step 5 --live
```

## 5. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `demo --scenario 2` behaves oddly on re-run | Reuse a fresh `--data-root` per run. The scenario seeds `crash-adder`; a second run on the same root does not start from a broken tree. |
| Successor gets `LeaseNotHeld` | You didn't wait out the crash TTL (`AGENT_CRASH_TTL`, default 2s). `sleep 3`, then retry — the successor only steals after expiry. |
| Worker A exits 0 instead of 137 | `AGENT_CRASH_AFTER` names the tool whose write triggers the kill (`replace_text` here). With `AGENT_CANNED=repair` the crash fires on the replace step too, but step 0 (read) will already be committed — same resume shape, one more committed row. |
| `UnknownTenant: acme` | Same rule as the main demo: every command must share one `--data-root`; an empty var silently resolves to the repo root. See `README_DEMO.md` §7. |
| Live scenario hangs | Slow endpoint (60s socket timeout per call). The crash scenario needs several model calls; if the endpoint stalls, the offline §2 recipe demonstrates the identical resume mechanics with no network. |

Reference: full setup + plan/apply/test flow in `README_DEMO.md`; framework gates in `README.md` §8; scenarios in `src/agent_engine/demo.py`; worker + hooks in `src/agent_engine/worker.py`.
