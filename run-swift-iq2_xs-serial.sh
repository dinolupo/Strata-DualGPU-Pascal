#!/bin/sh
# Profilo "serial": schede a turno (--pipeline-windows 0), KV tutto in VRAM, riserve 512/512 MiB,
# log del motore in logs/engine-serial.log. Pensato per le schede a piena potenza (250 W / 180 W):
# con le schede a turno i consumi non si sommano. Il servizio gpu-power-limit le abbassa a ogni avvio.
cd "/home/dino/proj/genAI/strata/Strata-DualGPU-Pascal" || exit 1
limits=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | paste -sd/)
case "$limits" in
  250.00/180.00) ;;
  *) echo "NOTA: limiti di potenza attuali $limits W (piena potenza = 250.00/180.00)."
     echo "      Per la piena potenza:  sudo nvidia-smi -i 0 -pl 250 && sudo nvidia-smi -i 1 -pl 180"
     echo "      Prima di tornare al profilo performance:  sudo systemctl restart gpu-power-limit" ;;
esac
exec "./.venv/bin/python" "./serve/server.py" "--engine" "strata" "--config" "./strata-swift-pascal-serial.json" "--port" "8080" "--open"
