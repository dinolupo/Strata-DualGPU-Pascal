#!/bin/sh
cd "/home/dino/proj/Strata-DualGPU-Pascal"
exec "/home/dino/proj/Strata-DualGPU-Pascal/.venv/bin/python" "/home/dino/proj/Strata-DualGPU-Pascal/serve/server.py" "--engine" "strata" "--config" "/home/dino/proj/Strata-DualGPU-Pascal/strata-swift-upstream-pascal.json" "--port" "8080" "--open"
