#!/usr/bin/env bash
# DELETES the demo SQLite databases, then replays the step 1 and step 2 walkthrough.
#
# Removed on every run:
#   /tmp/agent-mode-demo/tenants/*/engine.db
#   /tmp/agent-mode-selftest/tenants/*/engine.db
#
# This script does not touch ./var or any other data root. Read it before you run it.

set -euo pipefail

cd "$(dirname "$0")/.."

ROOT=/tmp/agent-mode-demo
SELFTEST=/tmp/agent-mode-selftest
A=11111111-1111-4111-8111-111111111111
B=22222222-2222-4222-8222-222222222222

if [[ ! -x .venv/bin/python ]]; then
  echo "missing .venv. From the repo root: python3 -m venv .venv && .venv/bin/python -m pip install -e '.[dev]'" >&2
  exit 1
fi

echo "DELETING demo databases:"
echo "  $ROOT"
echo "  $SELFTEST"
rm -rf "$ROOT" "$SELFTEST"

PY=.venv/bin/python

run() {
  echo
  echo "+ python -m agent_engine $*"
  "$PY" -m agent_engine "$@"
}

expect_fail() {
  local needle=$1
  shift
  echo
  echo "+ python -m agent_engine $*    (expected failure: $needle)"
  set +e
  local output
  output=$("$PY" -m agent_engine "$@" 2>&1)
  local code=$?
  set -e
  printf '%s\n' "$output"
  if [[ $code -eq 0 ]]; then
    echo "expected a failure containing $needle" >&2
    exit 1
  fi
  if [[ "$output" != *"$needle"* ]]; then
    echo "failure did not contain $needle" >&2
    exit 1
  fi
}

run --data-root "$ROOT" provision --tenant acme --display-name Acme
run --data-root "$ROOT" provision --tenant globex --display-name Globex
run --data-root "$ROOT" init-task --tenant acme --task fix-adder
run --data-root "$ROOT" init-task --tenant globex --task fix-adder
run --data-root "$ROOT" pragma --tenant acme

run --data-root "$ROOT" acquire --tenant acme --task fix-adder --worker-id "$A" --now 0 --ttl 10
run --data-root "$ROOT" show-lease --tenant acme --task fix-adder
run --data-root "$ROOT" renew --tenant acme --task fix-adder --worker-id "$A" --token 1 --now 5 --ttl 10
expect_fail LeaseNotHeld --data-root "$ROOT" acquire --tenant acme --task fix-adder --worker-id "$B" --now 5 --ttl 10
run --data-root "$ROOT" show-lease --tenant acme --task fix-adder
expect_fail LeaseNotHeld --data-root "$ROOT" acquire --tenant acme --task fix-adder --worker-id "$B" --now 11 --ttl 10
run --data-root "$ROOT" acquire --tenant acme --task fix-adder --worker-id "$B" --now 15 --ttl 10
run --data-root "$ROOT" show-lease --tenant acme --task fix-adder
run --data-root "$ROOT" show-lease --tenant globex --task fix-adder

run --data-root "$ROOT" release --tenant acme --task fix-adder --worker-id "$B" --token 2
run --data-root "$ROOT" show-lease --tenant acme --task fix-adder
run --data-root "$ROOT" acquire --tenant acme --task fix-adder --worker-id "$A" --now 1 --ttl 10
expect_fail FenceRejected --data-root "$ROOT" release --tenant acme --task fix-adder --worker-id "$B" --token 2

run self-test --through 2 --data-root "$SELFTEST"
expect_fail FAIL self-test --through 2 --data-root "$SELFTEST"

echo
echo "Done. Demo databases were deleted at the start and recreated under:"
echo "  $ROOT"
echo "  $SELFTEST"
