# Dual Pascal: certified build (dinolupo fork)

Copy of `bin/CERTIFIED.md` from the hub `~/proj/genAI/strata` (Gerty: GTX 1080 Ti + GTX 1070, 32 GB DDR3).
Profile: `strata-swift-pascal-v0.1.45.json`, launcher: `run-swift-v0.1.45.sh`. Porting notes: `PORT_PLAN.md`.

# Binari Strata in questa cartella

| File | Versione | Codice | sha256 | Stato |
|---|---|---|---|---|
| `strata-v0.1.45-steering` | 0.1.45 + steering portato | `port/steering-v0.1.45` @ `9959fa04` (C++ invariato da `21cd5f57`, build 18/10/2026 02:39) | `980280259bd3c484f44f…` | **CERTIFICATO 18/10/2026**, profilo quotidiano |
| `strata-v0.1.45-baseline` | 0.1.45 pulita | tag `v0.1.45` (`477ac13c`) | `6854cdc5be5c8d09…` | riferimento per i confronti |

Certificazione `strata-v0.1.45-steering`: `run-port-tests.sh`, log `logs/port-tests-20261018-030517.log`:
2 test CPU + `test_steering_e2e`, `test_steering_server`, `steering_ab` (layer 4-44, 128K) tutti PASS, 0 Xid.
Uso reale con pi.dev (tappa 2 del porting) senza blocchi del motore.

Server e tool da usare con questo binario: worktree congelato `Strata-v0.1.45-steering` (detached a `9959fa04`).
Non ricompilare sopra questi file: una nuova versione va in un file nuovo, testata e poi certificata qui.
Rollback: profilo del fork, `Strata-DualGPU-Pascal/run-swift-iq2_xs-performance.sh` (binario `build-upstream-pascal/strata`).
