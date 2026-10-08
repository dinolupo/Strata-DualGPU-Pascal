#!/bin/sh
cd "/home/dino/proj/genAI/strata/Strata-DualGPU-Pascal" || exit 1
exec "./.venv/bin/python" "./serve/server.py" "--engine" "strata" "--config" "./strata-swift-pascal-performance.json" "--port" "8080" "--open"
