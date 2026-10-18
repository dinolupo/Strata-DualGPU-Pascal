#!/bin/bash
# PROFILO QUOTIDIANO (dal 18/10/2026): Strata v0.1.45 + steering portato, binario certificato.
# Binario: bin/strata-v0.1.45-steering (vedi bin/CERTIFIED.md). Server e tool: worktree congelato
# Strata-v0.1.45-steering (commit 9959fa04, detached). Contesto 64K, 20 GB di esperti in RAM.
# Il vecchio profilo del fork resta in Strata-DualGPU-Pascal/run-swift-iq2_xs-performance.sh (rollback).
HUB=/home/dino/proj/genAI/strata
if pgrep -x strata >/dev/null || pgrep -f "strata-v0.1.45" >/dev/null || pgrep -f "serve/server.py" >/dev/null; then
    echo "ABORT: un altro Strata è acceso. Chiudilo e rilancia."; pgrep -af "strata|serve/server.py"; exit 1
fi
[ -x "$HUB/bin/strata-v0.1.45-steering" ] || { echo "ABORT: manca $HUB/bin/strata-v0.1.45-steering"; exit 1; }
nvidia-smi --query-gpu=index,name,power.limit --format=csv,noheader
echo "log del motore: $HUB/logs/engine-v0.1.45-steering.log"
cd "$HUB/Strata-v0.1.45-steering" || exit 1
exec "$HUB/Strata-DualGPU-Pascal/.venv/bin/python" ./serve/server.py --engine strata \
     --config "$HUB/strata-swift-pascal-v0.1.45.json" --port 8080 --open
