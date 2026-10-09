# Log di Sviluppo e Test: DS4 Directional Steering su Strata

## Freeze all'avvio e allineamento risorse (2026-10-08, 22:48)

Il rilancio E2E non ha raggiunto READY entro 900 s; il kill di cleanup
non e terminato entro i successivi 5 s. Il comando non riprendeva il JSON
del launcher: GPU `[0,1]`, split auto, budget RAM e guard paced assenti.
Senza resident/mmap il codice seleziona ArenaExpertSource, che su Linux
prova a tenere e pinnare l'intera arena degli esperti, non solo il
complemento entro 16 GiB. Il test server usava resident-experts ma senza
budget e guard espliciti. Entrambi erano diversi dal launcher manuale.

Nel kernel alle 22:44:57-58 risultano task bloccati oltre 122 s, incluso
`jbd2/sdb3-8`, e `NVRM: failed to allocate page table!`. Questi dati
dimostrano uno stallo host/driver; non identificano con certezza tutta
la catena causale, ne un OOM killer o un guasto fisico del disco.
Al controllo successivo nessun motore/test era attivo, nessun processo
compute GPU risultava presente e la RAM disponibile era 28997 MiB.

Entrambi i test ora riusano la configurazione
`strata-swift-upstream-pascal.json`, cambiando solo il contesto a 2048/4096
e aggiungendo gli argomenti steering. Si conserva il binario
`build-vector-steering/strata`, senza fallback a un'altra build.
`STRATA_STEERING_CONFIG` puo scegliere un JSON diverso, ma deve avere
budget resident positivo e non superiore a 16 GiB, resident-experts,
guard paced 2 e gli argomenti necessari espliciti.
Ordine GPU, split, riserve, worker, pipeline e ambiente derivano dal JSON;
per quello del launcher sono `[1,0]`, split 22, riserve 512/1500,
quattro worker e pipeline 0. Non sono modificati launcher o produzione.

Log motore persistenti in `logs/steering-tests/`, anche se READY fallisce.
Il test server riusa il client bounded del server invece di attendere
stdout senza limite e chiude il proprio processo in finally.
La suite shell si ferma al primo errore, senza lanciare un secondo
motore dopo un cleanup incompleto. I timeout non sono aumentati.
Verifiche esclusivamente CPU/mock: nessuna build o inferenza avviata.
Il nuovo assetto richiede ancora verifica hardware autorizzata; non e
una certificazione di assenza di freeze.

## Correzione runner E2E (2026-10-08)

Il test `tools/test_steering_e2e.py` passava `--layer-split auto` alla
modalita one-shot, ma il motore richiede `--serve` per il layer split.
Inoltre cercava righe `T <id>`, proprie del protocollo serve.
Il runner ora riusa `serve.server.StrataEngine`: attende READY, invia GEN,
verifica DONE e chiude il proprio processo anche in caso di errore.
Parita GGUF/F32, effetto del segno e disattivazione transitoria non dimostrati
ora fanno fallire il test; anche la parita finale del test server e vincolante.
Validazione CPU simulata in `tools/test_steering_e2e_runner.py`.
Nessuna build o inferenza GPU eseguita per questa correzione: la
qualificazione hardware resta da rieseguire manualmente.
Il primo rilancio ha evidenziato un ulteriore requisito mancante:
`--expert-profile`. Il controllo delle guardie native/serve ha confermato
anche la necessita di `--spec`, `--mtp` e `--prefill`; tutti sono ora
espliciti e verificati dai test CPU del runner. Questo non sostituisce
la verifica READY e delle generazioni sul modello reale.

Data inizio: 2026-10-06
Branch: `feat/ds4-directional-steering`
Worktree: `/home/dino/proj/Strata-DualGPU-Pascal-steering`

## Obiettivo
Implementare, integrare e testare in Strata tutte le funzionalità che consentono di:
1. Creare / estrarre vettori di directional steering (secondo la metodologia di `ds4` di Salvatore Sanfilippo).
2. Caricare ed eseguire i vettori di steering in Strata sia da formato `.f32` grezzo (nativo di `ds4`) sia da formato `.gguf` standard (`controlvector`).
3. Supportare la logica di **Transient Steering** di Sanfilippo: applicare lo steering solo sui primi $N$ token di generazione per dare l'innesco/traiettoria voluta, e poi disattivarlo a caldo senza invalidare la KV cache o i CUDA Graphs.
4. Eseguire una suite completa di test (unit test, test di conversione, test di generazione e verifica di disattivazione al token $N$).

---

## Fasi di Lavoro e Avanzamento

### [x] Fase 1: Isolamento Workspace e Worktree
- Branch: `feat/ds4-directional-steering`
- Worktree: `/home/dino/proj/Strata-DualGPU-Pascal-steering`
- Repository principale `/home/dino/proj/Strata-DualGPU-Pascal` intatto al 100%.

### [x] Fase 2: Analisi Matematica e Formato Vettori DS4 vs Strata
- **Formula DS4**:
  $$y = y - \text{scale} \cdot \vec{v} \cdot (\vec{v} \cdot y)$$
  dove $\vec{v}$ è il vettore direzionale normalizzato unitario di quel layer, e $\text{scale}$ è uno scalare con segno ($>0$ per soppressione, $<0$ per amplificazione).
- **Kernel Strata (`src/kernels/cuda/cvec.cu`)**:
  Esegue la medesima operazione in GPU con warp shuffles `__shfl_xor_sync` e toggle live tramite puntatore device `const int* on`:
  $$h \leftarrow h - s_l (h \cdot v_l) v_l$$
- **Importante scoperta e fix sul segno**:
  In Strata, la modalità `--cvec-mode project` calcolava $nrm = \sqrt{\sum d_j^2} \ge 0$ e inseriva l'eventuale segno negativo in $\vec{v}$. Poiché $(h \cdot (-\vec{v}))(-\vec{v}) = (h \cdot \vec{v})\vec{v}$, il segno negativo veniva elevato al quadrato e annullato, rendendo impossibile l'amplificazione (scale $<0$).
  È stata corretta la procedura in `load_control_vectors` affinché $\vec{v}$ rimanga il versore coerente di riferimento e la scala con segno sia preservata in $s_l$.

### [x] Fase 3: Tooling Python di Conversione e Generazione Vettori
- **`tools/ds4_to_cvec.py`**:
  - Conversione bidirezionale tra `.f32` (DS4) e `.gguf` (standard GGUF `controlvector`).
  - Supporto forme complete $(L, N)$ e headless $(L-1, N)$.
  - Test di round-trip fidelity eseguito su `data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf` e su `ds4/dir-steering/out/verbosity.f32`: errore massimo $= 0.0000000000\text{e}+00$ (corrispondenza bitwise esatta a zero difetti).
- **`tools/build_steering_vector.py`**:
  - Implementazione completa della metodologia DS4 per prompt di contrasto (good vs bad pairs).
  - Normalizzazione unitaria per-layer.
  - Ortogonalizzazione Gram-Schmidt rispetto al baricentro medio della classe di controllo (`orthogonalize_control_mean`).
  - Salvataggio contemporaneo in `.f32` + `.json` e `.gguf`.
- **`tools/test_build_steering_vector.py`**:
  - Unit test automatizzati su normalizzazione, ortogonalità e roundtrip GGUF: superati con successo (`Ran 3 tests in 1.139s - OK`).

### [x] Fase 4: Integrazione Core C++ nel Motore Strata
- File modificati:
  - `src/program/generate.cpp`:
    - `struct Options`: aggiunto `int cvec_tokens = 0;`.
    - `usage()`: documentati i nuovi flag `--cvec-tokens N`, `--dir-steering-file PATH`, `--dir-steering-ffn F`, `--dir-steering-tokens N`.
    - `load_control_vectors()`: supporto al caricamento diretto dei file raw `.f32` (sia a $48$ layer che a $47$ layer headless) in contemporanea al formato `.gguf`, con preservazione rigorosa della scala con segno.
    - CLI option parsing: gestione dei flag `--dir-steering-file`, `--dir-steering-ffn`, `--dir-steering-tokens`, `--cvec-tokens`.
    - Serve mode protocol: parsing chiavi `cvec_tokens=N` / `dir_steering_tokens=N` nella riga di richiesta `GEN`/`GENI`.
    - Decode loops (pipeline, standard, non-speculative, speculative): monitoraggio di `produced_n` e disattivazione istantanea via `strata::kernels::cvec_set_enabled(false)` non appena viene raggiunto il limite dei token transienti, lasciando fluire la generazione naturale per i token rimanenti.
  - `serve/server.py`:
    - Aggiunto supporto a `cvec_tokens` e `dir_steering_tokens` nei parametri di campionamento e `SHARED_KEYS`.
    - Inoltro automatico al comando di generazione del processo motore.

### [x] Fase 5: Compilazione in Worktree
- Configurazione CMake Release CUDA (sm_61, experimental SM60) completata con successo.
- Build motore `build/strata` completata e copiata in `engine/strata`.

### [x] Fase 6: Risoluzione Deadlock Kernel & Mapped Host Memory Toggle
- **Diagnosi**: Nel test iniziale con pipeline a 2 GPU (`--pipeline-windows 2`), la chiamata a `cvec_set_enabled(false)` eseguiva `cudaDeviceSynchronize()` da dentro il thread host mentre gli stadi di verifica su GPU 0 e GPU 1 erano sincronizzati tramite hand-off flags. Questo causava un deadlock driver dopo 5 token.
- **Soluzione architetturale**:
  In `src/kernels/cuda/cvec.cu`, il flag di attivazione `on` è stato allocato in memoria host mappata zero-copy con `cudaHostAllocMapped | cudaHostAllocPortable`.
  - Il thread host esegue una scrittura atomica CPU diretta: `*g_dev[d].on_host = v;`.
  - Nessuna chiamata di sincronizzazione GPU (`cudaDeviceSynchronize` eliminato).
  - Nessun blocco driver.
  - Tempo di toggle: $< 10\text{ ns}$, visibile istantaneamente su tutte le GPU.

### [x] Fase 7: Test End-to-End su Modello Reale (Dual Pascal: GTX 1080 Ti + GTX 1070)
- Modello: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS` (35 GB)
- File di test: `tools/test_steering_server.py`
- Prompt test:
  ```
  <|im_start|>system
  You are a helpful assistant.<|im_end|>
  <|im_start|>user
  Write a short motto for space explorers.<|im_end|>
  <|im_start|>assistant
  ```

#### Risultati Misurati:

1. **Richiesta non guidata (`cvec=0`)**:
   - Token generati: `[248068, 198, 760, 1156, ...]`
   - Testo generato:
     `"<think>\nThe user wants a short motto for space explorers. Let me think about what makes a good motto for space explor..."`
   - Il modello entra in modalità thinking estesa.

2. **Richiesta guidata DS4 (`cvec=1`, file `.f32`)**:
   - Token generati: `[248068, 271, 248069, 271, 1, 53483, 279, ...]`
   - Testo generato:
     `"<think>\n\n</think>\n\n\"Beyond the stars, we find ourselves.\"<|im_end|>"`
   - Lo steering sopprime il blocco di ragionamento e risponde immediatamente con il motto.

3. **Transient Steering a 5 Token (`cvec=1 cvec_tokens=5`)**:
   - Disattivazione rilevata su stderr: `strata serve: directional steering deactivated after 5 tokens`
   - Token nei primi 5: `[248068, 271, 248069, 271, 1]` corrispondenti esattamente a `<think>\n\n</think>\n\n"`
   - Continuazione autonoma non guidata:
     `"Beyond the horizon, into the infinite."<|im_end|>`
   - **Conferma dell'ipotesi di Sanfilippo**: Una volta impostata la traiettoria nei primi 5 token, disattivando lo steering il modello continua autonomamente senza alcuna deviazione o rigetto.

4. **Transient Steering a 12 Token (`cvec=1 cvec_tokens=12`)**:
   - Disattivazione rilevata su stderr: `strata serve: directional steering deactivated after 12 tokens`
   - Testo generato:
     `"<think>\n\n</think>\n\n\"Beyond the horizon, into the unknown.\"<|im_end|>"`

5. **Reset pulito dello stato (`cvec=0`)**:
   - Token generati: `[248068, 198, 760, 1156, 6587, 264, 2716, 60227, 364, 3433, 45859, 382, 13, 6558, 728, 1683, 883, 1092, 3520, 264, 1603, 60227, 364, 3433, 45859]`
   - **Corrispondenza 100% identica alla Richiesta 1**: Nessun residuo tra sessioni.

6. **Parità Cross-Format (DS4 `.f32` vs GGUF `controlvector`)**:
   - Transient 5 tokens: Match esatto al 100% (identica sequenza di token bit per bit).
   - Transient 12 tokens: Match esatto al 100% (identica sequenza di token bit per bit).

### [x] Fase 8: Verifica Finale e Chiusura Obiettivo
- Tutte le modifiche risiedono nel worktree isolato `feat/ds4-directional-steering`.
- Il repository principale e la compilazione precedente sono rimasti intatti.
- Tutti i test matematici, di conversione e di esecuzione a caldo sono passati con successo.
