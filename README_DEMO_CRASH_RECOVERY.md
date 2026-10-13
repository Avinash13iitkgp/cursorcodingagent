# Crash & Recovery on a Larger Change: Kill Mid-Repair, Successor Finishes It (Live)

Scenario 2 proves crash-restart on a **single** edit. This guide scales it up:
a task with **three broken functions across two files**, where worker A is
SIGKILLed on its first edit and a live successor resumes the open intent,
repairs the remaining files, and drives the task to **`completed`**.

No code changes needed. The trick is using the repair-oriented worker loop
(`python -m agent_engine.worker`, which owns the `AGENT_CRASH_AFTER` hook)
instead of `agent apply` (which has no crash hooks, and whose hardcoded 180s
TTL would make an external kill-and-resume wait out the lease).

## 0. Why this shape

- `AGENT_CRASH_AFTER=replace_text` fires on the **first** `replace_text` write,
  wherever it lands — so the crash point is order-independent.
- The successor runs the **full** live loop (not `AGENT_RESUME_ONLY`, which stops
  after the open intent): it replays the crashed intent as `already_applied`
  with zero writes, then fixes the remaining files, calls `run_checks`, calls
  `finish`, and the verifier writes `completed`.
- The default worker prompt ("fix `adder.py`") stays aligned by keeping the
  broken adder as one of the files; the failing `run_checks` output steers the
  model to the rest. `finish` cannot succeed until all tests pass, so the model
  cannot declare victory early.

## 1. Seed the multi-file broken task

`$BIG/checks/commands` **must** exist: without it the verifier falls back to the
adder-fixture default commands, which would pass vacuously on this tree. Tests
live under `base/tests/` so they are seeded into `work/`, where pytest collects
them with `PYTHONPATH` pointed at the work root.

```bash
export BIG=/tmp/agent-crash-big
rm -rf "$BIG"
agent --data-root "$BIG" provision --tenant acme --display-name Acme
agent --data-root "$BIG" init-task --tenant acme --task crash-big
mkdir -p "$BIG/tenants/acme/tasks/crash-big/base/tests" "$BIG/checks"
cp fixtures/broken_adder/adder.py "$BIG/tenants/acme/tasks/crash-big/base/adder.py"
printf 'python -m pytest -q\n' > "$BIG/checks/commands"
```

Write these two files into `base/`:

```python
# $BIG/tenants/acme/tasks/crash-big/base/calc.py — both functions intentionally wrong
"""Two-function calculator. Both are intentionally wrong."""

def mul(a: int, b: int) -> int:
    return a + b


def sub(a: int, b: int) -> int:
    return a + b
```

```python
# $BIG/tenants/acme/tasks/crash-big/base/tests/test_all.py — correct tests, all three fixes
from adder import add
from calc import mul, sub


def test_add() -> None:
    assert add(2, 3) == 5


def test_mul() -> None:
    assert mul(2, 3) == 6


def test_sub() -> None:
    assert sub(2, 3) == -1
```

Then seed `work/` from `base/` (hardlinks; `base/` is never written afterward):

```bash
python -c "
from pathlib import Path
from agent_engine.adapters.workspace.cow import CowWorkspace
CowWorkspace(Path('$BIG/tenants/acme/tasks/crash-big')).seed()
"
```

## 2. Crash worker A on its first edit (live)

```bash
export AGENT_LLM_BASE_URL="https://api.openai.com/v1"
export AGENT_LLM_API_KEY="the token you provide"
export AGENT_LLM_MODEL="the model name for that endpoint"

AGENT_CRASH_AFTER=replace_text AGENT_CRASH_TTL=2 \
  python -m agent_engine.worker --data-root "$BIG" --tenant acme --task crash-big \
  --llm live --ttl 180
echo "worker A exit=$?"   # 137 = SIGKILL
```

Inspect the post-crash state — the shape to expect (values observed on the
single-file run; here the crashed step is whichever file the model fixed first):

```bash
grep -rn 'return a' "$BIG/tenants/acme/tasks/crash-big/work/adder.py" "$BIG/tenants/acme/tasks/crash-big/work/calc.py"
python -c "
import sqlite3
c = sqlite3.connect('file:$BIG/tenants/acme/engine.db?mode=ro', uri=True)
print('TASK:', c.execute(\"SELECT id, status, fencing_token FROM tasks WHERE id='crash-big'\").fetchall())
print('STEPS:', c.execute(\"SELECT step_index, tool_name, status FROM step_intents WHERE task_id='crash-big'\").fetchall())
"
```

Expect: one file fixed on disk, task `running` at fence 1, and the crashed step
sitting at `intent_logged` (T1 durable, T2 never ran).

## 3. Recover with a full live successor

Wait out the 2s crash lease, then run the successor **without** `AGENT_RESUME_ONLY`
so it keeps going after the open intent:

```bash
sleep 5
python -m agent_engine.worker --data-root "$BIG" --tenant acme --task crash-big \
  --llm live --ttl 180 --max-steps 100
echo "successor exit=$?"   # 0
```

`--max-steps 100` gives the successor room for the whole repair in one run
(worker default is 8; `run-task`/`apply`/`new` already had this flag).

Expected trajectory: the open intent replays as `already_applied` (zero writes)
→ the model reads and fixes the remaining broken functions → `run_checks`
reports `3 passed` → `finish` → verifier writes **`completed`**.

Verify the recovery:

```bash
python -c "
import sqlite3
c = sqlite3.connect('file:$BIG/tenants/acme/engine.db?mode=ro', uri=True)
print('TASK:', c.execute(\"SELECT id, status, fencing_token FROM tasks WHERE id='crash-big'\").fetchall())
print('STEPS:', c.execute(\"SELECT step_index, tool_name, status FROM step_intents WHERE task_id='crash-big'\").fetchall())
"
cd "$BIG/tenants/acme/tasks/crash-big/work" && PYTHONPATH=. python -m pytest -q
```

Expect: `status=completed`, every step `committed`, fencing token ≥ 2, and
`3 passed` locally.

> Budget note: each worker run allows **8 steps by default**. On a larger tree the model
> can spend them on exploration (`list_files`, `search_text`, extra reads) plus
> the first fixes and stop with the task still `running` — exactly what happened
> here (steps 0–7 used, `calc.py` untouched). That is not a failure: just **re-run
> the §3 successor command** (now with `--max-steps 100`, so one run usually
> suffices). Committed steps are never re-executed, so a repeat run resumes from
> step 8 with a fresh budget and continues where the last one stopped. Repeat until
> `status=completed`.

## 4. What this proves (talk track)

| Fact | Evidence |
| --- | --- |
| Kill loses no work | Crashed step stays `intent_logged`; its bytes are already on disk |
| Successor never rewrites | Open intent replays as `already_applied`, zero writes for that step |
| Recovery finishes the job, not just one commit | Remaining files fixed by the same successor; task reaches `completed` |
| Fence integrity across the kill | Token 1 (dead) → 2 (successor lease) → +1 per clean release |
| Operator gate still owns the outcome | `finish` → verifier re-runs `checks/commands` → `completed` |

## 5. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `run_checks` passes without testing anything | `checks/commands` is missing — the verifier fell back to the adder-fixture default. Re-create it per §1 and re-seed on a fresh root. |
| Successor gets `LeaseNotHeld` | Crash TTL not yet expired. `sleep 5` covers the default 2s `AGENT_CRASH_TTL`. |
| Successor stops after the open intent | You set `AGENT_RESUME_ONLY=1` (right for the single-edit demo, wrong here). Unset it so the full loop continues. |
| `2 failed, 1 passed` after the successor (calc still broken, task `running`) | Per-run 8-step budget exhausted (check: 8 rows in `step_intents`, all `committed`). Re-run the §3 successor command — it resumes from the next step with a fresh budget. Repeat to `completed`. |
| Re-run behaves oddly | Fresh `--data-root` per attempt. A completed tree has nothing left to crash on. |
| Pytest import error on local verification | Run from the work root with `PYTHONPATH=.` (`from adder import add` needs the work root importable). |
| Everything is slow | Each model turn is one live call against your endpoint. Expect ~8–12 turns; the crash lands after 2–4, the successor's repair is the long pole. |

Reference: single-edit crash cycle in `README_DEMO_CRASH_RESTART.md` §2 (fully verified
output included); suspend/steal + gates in §3–§4 there; main build/test flow in
`README_DEMO.md`; worker + hooks in `src/agent_engine/worker.py`; check resolution
in `src/agent_engine/adapters/verifier/external.py`.
