# Strata Pascal: rapporto dei test notturni

Build candidata: backport selettivo upstream, non merge completo.

## Esito

Conteggi: {'PASS': 19, 'FAIL': 4, 'ERROR': 4, 'BLOCKED_GPU_OCCUPIED': 1, 'INCOMPLETE_EVIDENCE': 1, 'NOT_RUN': 32}.
Nessuna configurazione e' qualificata se mancano soak, conversazione e parity.
Casi con evidenza incompleta: ['context-16384']. Non sono PASS; l'interruzione non e' attribuita al deadline senza prova.
Il prompt massimo nei casi PASS e' 24576 token. La tabella distingue casi completati, interrotti e mai avviati.

| Caso | Stato | Richieste | Prompt massimo reale | Risposte corrette | Cancellazioni |
|---|---|---:|---:|---:|---:|
| thinking-1024 | PASS | 3 | 75 | 3 | 0 |
| thinking-2048 | PASS | 3 | 75 | 3 | 0 |
| thinking-4096 | PASS | 3 | 75 | 3 | 0 |
| thinking-long | PASS | 3 | 24576 | 3 | 0 |
| stable-controlled | FAIL | 1 | 35 | 1 | 0 |
| candidate-controlled | PASS | 3 | 8192 | 3 | 0 |
| reverse-order | PASS | 3 | 8192 | 3 | 0 |
| serial | PASS | 3 | 8192 | 3 | 0 |
| adaptive-off | FAIL | 3 | 8192 | 3 | 0 |
| adaptive-blocking | ERROR | 1 | 35 | 1 | 0 |
| thinking-forced-wrap | PASS | 3 | 75 | 3 | 0 |
| reasoning-arithmetic | PASS | 3 | 165 | 3 | 0 |
| catchup-all | FAIL | 1 | 35 | 1 | 0 |
| stager-yield | ERROR | 1 | 35 | 1 | 0 |
| split20 | BLOCKED_GPU_OCCUPIED | 0 | 0 | 0 | 0 |
| http-thinking | PASS | 3 | 75 | 3 | 0 |
| reverse-thinking | PASS | 3 | 75 | 3 | 0 |
| serial-thinking | PASS | 3 | 75 | 3 | 0 |
| split24 | FAIL | 1 | 35 | 1 | 0 |
| resident18 | PASS | 3 | 8192 | 3 | 0 |
| resident20 | PASS | 3 | 8192 | 3 | 0 |
| stable-explicit-off | ERROR | 1 | 35 | 1 | 0 |
| candidate-explicit-off | PASS | 3 | 8192 | 3 | 0 |
| pipeline1 | ERROR | 1 | 35 | 1 | 0 |
| serial-spec2 | PASS | 3 | 8192 | 3 | 0 |
| adaptive-off-checked | PASS | 3 | 8192 | 3 | 0 |
| adaptive-blocking-retry | PASS | 3 | 8192 | 3 | 0 |
| context-4096 | PASS | 3 | 4096 | 3 | 0 |
| context-16384 | INCOMPLETE_EVIDENCE | 1 | 16384 | 1 | 0 |
| context-24576 | NOT_RUN | 0 | 0 | 0 | 0 |
| context-30000 | NOT_RUN | 0 | 0 | 0 | 0 |
| conversation-reuse | NOT_RUN | 0 | 0 | 0 | 0 |
| serial-context30k | NOT_RUN | 0 | 0 | 0 | 0 |
| serial-conversation | NOT_RUN | 0 | 0 | 0 | 0 |
| reverse-context30k | NOT_RUN | 0 | 0 | 0 | 0 |
| reverse-conversation | NOT_RUN | 0 | 0 | 0 | 0 |
| soak100 | NOT_RUN | 0 | 0 | 0 | 0 |
| serial-soak100 | NOT_RUN | 0 | 0 | 0 | 0 |
| reverse-soak100 | NOT_RUN | 0 | 0 | 0 | 0 |
| what-fits | NOT_RUN | 0 | 0 | 0 | 0 |
| performance-0-stable | NOT_RUN | 0 | 0 | 0 | 0 |
| performance-0-candidate | NOT_RUN | 0 | 0 | 0 | 0 |
| performance-1-stable | NOT_RUN | 0 | 0 | 0 | 0 |
| performance-1-candidate | NOT_RUN | 0 | 0 | 0 | 0 |
| performance-2-stable | NOT_RUN | 0 | 0 | 0 | 0 |
| performance-2-candidate | NOT_RUN | 0 | 0 | 0 | 0 |
| performance-3-stable | NOT_RUN | 0 | 0 | 0 | 0 |
| performance-3-candidate | NOT_RUN | 0 | 0 | 0 | 0 |
| performance-4-stable | NOT_RUN | 0 | 0 | 0 | 0 |
| performance-4-candidate | NOT_RUN | 0 | 0 | 0 | 0 |
| order-performance-0-normal | NOT_RUN | 0 | 0 | 0 | 0 |
| order-performance-0-reverse | NOT_RUN | 0 | 0 | 0 | 0 |
| order-performance-1-normal | NOT_RUN | 0 | 0 | 0 | 0 |
| order-performance-1-reverse | NOT_RUN | 0 | 0 | 0 | 0 |
| order-performance-2-normal | NOT_RUN | 0 | 0 | 0 | 0 |
| order-performance-2-reverse | NOT_RUN | 0 | 0 | 0 | 0 |
| context-49152 | NOT_RUN | 0 | 0 | 0 | 0 |
| context-60000 | NOT_RUN | 0 | 0 | 0 | 0 |
| context-98304 | NOT_RUN | 0 | 0 | 0 | 0 |
| context-120000 | NOT_RUN | 0 | 0 | 0 | 0 |
| context-240000 | NOT_RUN | 0 | 0 | 0 | 0 |

## Raccomandazione

Configurazione qualificata: None.
Se il valore e' None, nessuna configurazione ha superato tutti i gate: non viene proposta come affidabile ne' installata.
Budget iniziale da provare: reasoning 1024, riserva risposta 2048, max_tokens almeno 3328; con contesti quasi pieni la richiesta viene rifiutata invece di esaurire lo spazio durante il thinking. Nessuna garanzia per risposte arbitrarie.
Dati numerici, velocita', configurazioni effettive e errori sono in report.json e nei file per caso.
stable-controlled usava i graph branches predefiniti del vecchio engine: non e' un confronto equo delle prestazioni.
adaptive-off: prima esecuzione fallita per un controllo errato del runner sul salvataggio del profilo, non per la risposta del modello. adaptive-off-checked usa il controllo corretto.
adaptive-blocking iniziale: interrotto dal timeout del monitor GPU; cleanup eseguito sui soli processi del test. Non e' un PASS.
I budget 1024/2048/4096 possono non venire raggiunti dai quesiti semplici. Il wrap forzato verifica il secondo passaggio reale dell'engine.
I contesti oltre 32K usano KV streaming, non KV interamente residente.
I gate sintetici non garantiscono correttezza per conversazioni arbitrarie.
La modalita' seriale con GPU fisiche [1,0] e pipeline 0 e' solo un candidato per ulteriori prove se non supera tutti i gate: non un fallback gia' certificato. Non vietare configurazioni sulla sola base degli errori del monitor/runner: valutare separatamente le ripetizioni con nome distinto.
Non ridurre max_tokens sotto il minimo necessario: il server rifiuta esplicitamente lo spazio insufficiente. La riserva non garantisce il completamento di ogni risposta e non corregge gli stalli CUDA.

## Protezioni

Tutti i file protetti invariati: True.

## Errori e prove incomplete

- stable-controlled: FAIL; EngineDied: the engine stopped unexpectedly (exit code -6)
- adaptive-off: FAIL; saved profile shape mismatch: None
- adaptive-blocking: ERROR; GPU monitor TimeoutExpired after 15 s; worker group cleaned up; driver responsive after cleanup
- catchup-all: FAIL; EngineDied: the engine stopped unexpectedly (exit code -6)
- stager-yield: ERROR; TimeoutExpired: Command '['nvidia-smi', '--query-gpu=index,memory.used,memory.free,temperature.gpu,utilization.gpu', '--format=csv,noheader,nounits']' timed out after 15 seconds
- split20: BLOCKED_GPU_OCCUPIED; vedere il log supervisor/engine
- split24: FAIL; EngineDied: the engine stopped unexpectedly (exit code -6)
- stable-explicit-off: ERROR; TimeoutExpired: Command '['nvidia-smi', '--query-gpu=index,memory.used,memory.free,temperature.gpu,utilization.gpu', '--format=csv,noheader,nounits']' timed out after 15 seconds
- pipeline1: ERROR; TimeoutExpired: Command '['nvidia-smi', '--query-gpu=index,memory.used,memory.free,temperature.gpu,utilization.gpu', '--format=csv,noheader,nounits']' timed out after 15 seconds
- context-16384: INCOMPLETE_EVIDENCE; No completed summary record; saved supervisor status: RUNNING. Execution interrupted or not finalized; not validated
- context-24576: NOT_RUN; No completed supervisor record; not validated
- context-30000: NOT_RUN; No completed supervisor record; not validated
- conversation-reuse: NOT_RUN; No completed supervisor record; not validated
- serial-context30k: NOT_RUN; No completed supervisor record; not validated
- serial-conversation: NOT_RUN; No completed supervisor record; not validated
- reverse-context30k: NOT_RUN; No completed supervisor record; not validated
- reverse-conversation: NOT_RUN; No completed supervisor record; not validated
- soak100: NOT_RUN; No completed supervisor record; not validated
- serial-soak100: NOT_RUN; No completed supervisor record; not validated
- reverse-soak100: NOT_RUN; No completed supervisor record; not validated
- what-fits: NOT_RUN; No completed supervisor record; not validated
- performance-0-stable: NOT_RUN; No completed supervisor record; not validated
- performance-0-candidate: NOT_RUN; No completed supervisor record; not validated
- performance-1-stable: NOT_RUN; No completed supervisor record; not validated
- performance-1-candidate: NOT_RUN; No completed supervisor record; not validated
- performance-2-stable: NOT_RUN; No completed supervisor record; not validated
- performance-2-candidate: NOT_RUN; No completed supervisor record; not validated
- performance-3-stable: NOT_RUN; No completed supervisor record; not validated
- performance-3-candidate: NOT_RUN; No completed supervisor record; not validated
- performance-4-stable: NOT_RUN; No completed supervisor record; not validated
- performance-4-candidate: NOT_RUN; No completed supervisor record; not validated
- order-performance-0-normal: NOT_RUN; No completed supervisor record; not validated
- order-performance-0-reverse: NOT_RUN; No completed supervisor record; not validated
- order-performance-1-normal: NOT_RUN; No completed supervisor record; not validated
- order-performance-1-reverse: NOT_RUN; No completed supervisor record; not validated
- order-performance-2-normal: NOT_RUN; No completed supervisor record; not validated
- order-performance-2-reverse: NOT_RUN; No completed supervisor record; not validated
- context-49152: NOT_RUN; No completed supervisor record; not validated
- context-60000: NOT_RUN; No completed supervisor record; not validated
- context-98304: NOT_RUN; No completed supervisor record; not validated
- context-120000: NOT_RUN; No completed supervisor record; not validated
- context-240000: NOT_RUN; No completed supervisor record; not validated

## Verifiche finali e risorse

Metadati delle verifiche effettivamente eseguite; nessun test mancante viene conteggiato come superato.

```json
{
  "checked_unix": 1791439717.9222832,
  "python_selected_suite": {
    "passed": 149,
    "failed": 0
  },
  "report_tests_final": {
    "passed": 12,
    "failed": 0
  },
  "ctest_compiled_selection": {
    "passed": 9,
    "failed": 0
  },
  "ctest_unfiltered_attempt": {
    "passed": 9,
    "not_run_missing_executables": 36,
    "exit_code": 8,
    "scope_error": "Unfiltered CTest includes targets not built in the separate candidate build; no full-suite success claimed"
  },
  "diff_check": "PASS",
  "gpu_compute_pids": [],
  "active_test_pids": [],
  "mem_available_kib": 28876952,
  "swap_used_kib": 0,
  "interrupted_case": "context-16384",
  "interruption_cause": "Confirmed unclean host restart: new boot at 04:02:05 CEST, last worker sample 04:01:27. Underlying restart cause remains unknown. See reboot-diagnosis.json.",
  "production_installed": false
}
```

## Riavvio della macchina

Nuovo boot: 2026-10-08T04:02:05+02:00. Ultimo campione salvato: 2026-10-08T04:01:27+02:00.
Riavvio non ordinato confermato: ha interrotto il test, non il limite delle sette ore. La causa del reset non e' provata. Nessun errore MCE/EDAC, OOM, panic o NVIDIA Xid trovato nel journal disponibile.
Temperature GPU registrate al massimo 70/59 C; temperatura CPU non registrata. Overclock CPU/RAM, alimentazione e un crash kernel/driver non registrato restano ipotesi.
Dettagli e limiti dell'evidenza in reboot-diagnosis.json.
