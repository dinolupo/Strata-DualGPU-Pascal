#!/bin/bash
# steering-test-run.sh — full e2e qualification of vector steering on feat/vector-steering.
# Stops the current Strata instance, runs the steering test suites, logs everything,
# then restarts Strata with the current (performance) configuration and verifies it.
#
# Usage:  ./steering-test-run.sh
# Log:    steering-tests-<timestamp>.log (path printed at the end)

cd "$(dirname "$0")" || exit 1
LOG="steering-tests-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee "$LOG") 2>&1

PY=./.venv/bin/python
LAUNCHER=./run-swift-iq2_xs-performance.sh
PASS=0; FAIL=0

say()  { echo "=== $*"; }
step() { echo; echo "###### $* ######"; }

step "START (log: $LOG)"

# 1. Stop the current Strata instance (server + engine launched by the launcher)
step "STOP current Strata instance"
pkill -f "serve/server.py" 2>/dev/null
pkill -f "build-upstream-pascal/strata" 2>/dev/null
say "waiting for GPU memory to be released..."
for i in $(seq 1 60); do
    FREE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    say "GPU used: ${FREE} MiB"
    [ "$FREE" -lt 1500 ] && break
    sleep 5
done

# 2. End-to-end steering test (engine runs: GGUF vs .f32 parity, signed scale, transient deactivation)
step "test_steering_e2e.py"
"$PY" tools/test_steering_e2e.py
RC=$?
say "test_steering_e2e.py exit=$RC"
[ $RC -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# 3. Server-level steering test (per-request cvec_tokens plumbing through server.py)
step "test_steering_server.py"
"$PY" tools/test_steering_server.py
RC=$?
say "test_steering_server.py exit=$RC"
[ $RC -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# 4. Restart Strata with the current configuration
step "RESTART Strata (current performance config)"
nohup "$LAUNCHER" > "strata-restart-$(date +%Y%m%d-%H%M%S).log" 2>&1 &
say "launcher pid=$!"

# 5. Verify the restart: port 8080 must answer within 120 s
step "VERIFY restart on port 8080"
UP=0
for i in $(seq 1 24); do
    sleep 5
    if curl -s -m 3 "http://127.0.0.1:8080/v1/models" | grep -q "data"; then
        UP=1; say "server is UP after $((i*5)) s"; break
    fi
    say "not answering yet ($((i*5)) s)"
done
[ $UP -eq 1 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

step "SUMMARY: PASS=$PASS FAIL=$FAIL — log: $(pwd)/$LOG"
[ $FAIL -eq 0 ] && say "ALL GREEN: steering qualified e2e, Strata restarted with current config."
[ $FAIL -gt 0 ] && say "SOMETHING FAILED: send this log to the agent for diagnosis."
exit $FAIL
