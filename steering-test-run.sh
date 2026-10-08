#!/bin/bash
# steering-test-run.sh — e2e qualification of vector steering on feat/vector-steering.
# Refuses to start if a Strata instance is active. YOU stop and restart Strata yourself.
#
# Usage:  ./steering-test-run.sh   (with the GPUs free)
# Log:    steering-tests-<timestamp>.log (path printed at the end)

cd "$(dirname "$0")" || exit 1
LOG="steering-tests-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee "$LOG") 2>&1

PY=./.venv/bin/python
PASS=0; FAIL=0

say()  { echo "=== $*"; }
step() { echo; echo "###### $* ######"; }

# 0. Guard: refuse to start while a Strata instance is active
if pgrep -f "serve/server.py" > /dev/null || pgrep -f "strata --serve" > /dev/null; then
    say "ABORT: a Strata instance is running. Stop it first, then re-run this script."
    pgrep -af "serve/server.py|strata --serve"
    exit 1
fi

step "START (log: $LOG)"

# 1. End-to-end steering test (engine runs: GGUF vs .f32 parity, signed scale, transient deactivation)
step "test_steering_e2e.py"
"$PY" tools/test_steering_e2e.py
RC=$?
say "test_steering_e2e.py exit=$RC"
[ $RC -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# 2. Server-level steering test (per-request cvec_tokens plumbing through server.py)
step "test_steering_server.py"
"$PY" tools/test_steering_server.py
RC=$?
say "test_steering_server.py exit=$RC"
[ $RC -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

step "SUMMARY: PASS=$PASS FAIL=$FAIL — log: $(pwd)/$LOG"
[ $FAIL -eq 0 ] && say "ALL GREEN: steering qualified e2e. Restart Strata yourself."
[ $FAIL -gt 0 ] && say "SOMETHING FAILED: send this log to the agent for diagnosis."
exit $FAIL
