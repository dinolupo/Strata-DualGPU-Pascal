# Studio: Directional Steering di ds4 (antirez) e l'implementazione Strata

Documento di studio per l'uso e la generazione di vettori direzionali su
Qwen3.8 Flash Next. Basato sull'analisi diretta del codice ds4
(`ds4.c`, `ds4_cuda.cu`, `dir-steering/tools/build_direction.py`) e del
codice Strata di questo repository (`src/kernels/cuda/cvec.cu`,
`src/program/generate.cpp`, `tools/build_steering_vector.py`), non solo dai README.

---

## 1. Che cos'è il vettore direzionale

Una matrice piatta `f32` di **vettori unitari normalizzati**, uno per ogni
layer transformer "normale":

- **Qwen3.8 Flash Next**: `48 × 2560` → 48·2560·4 byte = **491.520 byte**
  (esattamente la dimensione di `data/experimental-speed-projection/Qwen3.8-Flash-Next-steering.f32`).
- Il layer MTP predittore è **escluso**: in ds4
  `directional_steering_layer_count() = N_LAYER − N_NEXTN_PREDICT`
  (`ds4.c:998`). I draft MTP non sono steered; vengono verificati dal trunk
  steered, quindi `--mtp` resta compatibile.

## 2. Come si genera un vettore (metodo ds4, verificato nel codice)

`dir-steering/tools/build_direction.py`:

1. Per ogni coppia di prompt `(good, bad)` lancia il modello con `-n 1` e
   abilita il dump della riga **`ffn_out`** (uscita FFN) dell'**ultimo token
   del prompt**, per ogni layer (`DS4_METAL_GRAPH_DUMP_PREFIX`,
   `DS4_METAL_GRAPH_DUMP_NAME=ffn_out`, `POS=0`).
2. `direction[layer] = normalize(mean(tutte good) − mean(tutte bad))`
   (media globale), oppure con `--pair-normalize`: normalizza la differenza
   di ogni coppia prima di mediare.
3. **Ortogonalizzazione di Gram-Schmidt** (default ON, `--no-orthogonalize`
   la disattiva): rimuove dal vettore la componente parallela alla media dei
   control (`bad`), rinormalizza. Motivo: il vettore deve catturare solo la
   *differenza* target−control, non il contesto condiviso dei prompt.
4. Scrive un JSON di metadati (`format: ds4-directional-steering-v1`, shape,
   component, provenienza) e il `.f32` piatto.

## 3. Semantica a runtime

L'edit applicato a ogni layer, su ogni riga (token) del tensore:

```
y = y − scale · dir[layer] · dot(dir[layer], y)
```

- `scale > 0`: **sopprime** la direzione catturata (il comportamento "good"
  viene rimosso dal flusso residuo).
- `scale < 0`: **amplifica** la direzione catturata.
- `scale = 0` o file assente: percorso normale, nessuna modifica.
- In ds4 i default sono `--dir-steering-ffn 1` (col file), `--dir-steering-attn 0`.
- Il kernel (`directional_steering_project_kernel`, `ds4_cuda.cu:12451`)
  calcola il dot con riduzione per riga e sottrae `scale·dot·dir`.

## 4. Dove agisce su Qwen3.8 Flash Next (ds4.c:58058-58068)

- **FFN steering**: proietta il tensore **R** (flusso residuo) su **tutte le
  4 branche hyperconnection** (`rows = T × DS4_N_HC`) subito dopo il FFN di
  ogni layer, prima dell'espansione HC.
- **Attention steering** (ds4-only): proietta l'output attention (`blk`)
  prima dell'iniezione nel residuo. Default 0; il README ds4 lo definisce
  "fragile" e sperimentale.
- In **Strata** il kernel `cvec` applica la proiezione sul **folded write del
  layer** (il residuo, tutte le stream HC): è l'equivalente del lato FFN.
  **Il lato attention non è implementato** (il README di
  `dir-steering/README.md` lo dichiara esplicitamente come DS4-only).

## 5. Perché "alcuni modelli si bloccano verso la fine"

Il vettore è una direzione a basso rango catturata da prompt corti. Applicato
**continuamente** su output lunghi:

- dopo ~20-40 token: **over-saturazione stilistica** → ripetizione di
  pattern, deriva grammaticale, perdita di contenuto fattuale;
- scale negativi forti (−2, −3): collasso in ripetizione su alcuni prompt
  (documentato nel README ds4 con esempi reali);
- la direzione catturata è più affidabile nei layer centrali/FFN, dove
  stile, topic e comportamento sono rappresentati; l'attention steering è
  più instabile perché agisce su un segnale più "grezzo".

Da qui il **Transient Steering** (`--dir-steering-tokens N` /
`--cvec-tokens N`): il vettore agisce solo sui primi N token generati, poi
viene disarmato. In Strata il disarmo avviene via flag in host memory
zero-copy (`cudaHostAlloc mapped`), senza `cudaDeviceSynchronize` né
contese sul driver, quindi non rompe la pipeline dual-GPU. Verificato:
disarmo esatto al token 5/12/20, log `deactivated after N tokens`,
nessuno stallo, reset pulito (baseline successiva bitwise identica).

## 6. Il vettore attuale nel repository

`Qwen3.8-Flash-Next-steering.f32` e
`Qwen3.8-Flash-Next-experimental-speed-projection.gguf` **non derivano da
coppie di prompt**: sono la "speed projection" sperimentale di ligna
llama.cpp convertita in formato ds4 (`"note": "converted from GGUF
controlvector format"`). Serve per testare il meccanismo, non un comportamento
semantico specifico.

## 7. Generare un proprio vettore con Strata

`tools/build_steering_vector.py` accetta `.parquet`, `.jsonl`, `.txt`
(i parquet di `mlabonne/` vengono letti per colonna `text`/`prompt` senza
stampare testi, come da regole AGENTS.md):

```bash
cd ~/proj/genAI/strata/Strata-DualGPU-Pascal
./.venv/bin/python tools/build_steering_vector.py \
  --good-file ../mlabonne/harmless_alpaca/data/train-00000-of-00001.parquet \
  --bad-file  ../mlabonne/harmful_behaviors/data/train-00000-of-00001.parquet \
  --max-prompts 50 \
  --out out/concept-steering \
  --engine build-vector-steering/strata \
  --pack /home/dino/proj/Strata-data/packs/swift-iq2_xs \
  --ple-gguf /home/dino/proj/genAI/models/Qwen3.8-Flash-Next-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
```

Protocollo di studio degli effetti (sweep): stesso prompt, scale
`0, ±0.5, ±1, ±2`. Regola ds4: iniziare tra −1 e +2; se compaiono
ripetizione o perdita di contenuto, la scala è troppo forte. `tools/steering_ab.py`
implementa già il pattern (baseline via `cvec=0`, −1, +2, transient).

---

## 8. La differenza metodologica (e cosa succede senza la modifica)

**ds4 cattura `ffn_out`** (uscita FFN, prima dell'iniezione HC).
**Strata `--dump-layers` cattura `R` dopo ogni layer**, mediato sulle 4
stream HC (`build_steering_vector.py:175-188`). Non sono lo stesso tensore.

### Se NON si implementa il dump fedele di `ffn_out` in Strata

1. **Coerenza interna**: la cattura su R-post-layer e l'applicazione sul
   folded write (R) avvengono nello **stesso spazio**. Un vettore generato
   da Strata è autocoerente: cattura e applica la stessa geometria. Il
   metodo funziona e produce direzioni valide.
2. **Contaminazione della direzione**: R-post-layer contiene FFN **più**
   il contributo attention **più** il residuo accumulato dai layer
   precedenti. La direzione `good−bad` risulta più "mista": il rapporto
   segnale/rumore per il solo comportamento FFN è inferiore. Effetti attesi:
   scale effettive leggermente diverse dalle previste, effetti collaterali
   (stile, topic) più ampi, sweep da calibrare empiricamente.
3. **Non intercambiabilità con i vettori ds4**: i `.f32` costruiti da ds4
   (spazio `ffn_out`) applicati in Strata (spazio R) sono un'approssimazione
   — è esattamente ciò che accade oggi con il vettore speed-projection, che
   ha superato i test di parità di *formato* e di *meccanismo*, ma la cui
   semantica è ereditata da un'altra geometria.
4. **Confronto con la comunità**: un vettore generato da Strata non è
   confrontabile con uno generato da ds4 per gli stessi prompt pair; la
   riproducibilità degli esperimenti antirez richiede il dump `ffn_out`.

### Se si implementa il dump `ffn_out` fedele (modifica suggerita)

- Piccola estensione del grafo CUDA di Strata: esportare il tensore FFN-out
  per layer nel dump (il kernel esiste già nel percorso del folded write);
  `build_steering_vector.py` leggerebbe righe `ffn_out` come `build_direction.py`.
- Vettori Strata → semanticamente identici a ds4 per le stesse coppie di
  prompt; intercambiabilità dei `.f32` tra i due motori.
- Costo: modifica al grafo di inferenza (zona critica: CUDA graph capture,
  pipeline dual-GPU), da qualificare con la suite e2e esistente.

## 9. Stato di qualifica attuale (branch `feat/vector-steering`)

- Unit: matematica vettori, round-trip `.f32`↔`.gguf` bitwise, 9 regression
  test CPU sull'harness.
- Kernel: `cvec_parity` (on/off, add, project, signed scale).
- E2E: parità bitwise `.f32`↔GGUF su output reali, transient 5/12, reset
  pulito.
- A/B comportamentale: ffn=−1 e ffn=+2 differiscono da baseline e tra loro
  su 3 prompt; contesto 131072 con disarmo al token 20 (7,4 s, 117 token),
  evento nel log, nessun `Xid`.
- **Aperto**: steering su conversazione lunga reale (~58K prompt), stabilità
  nel tempo, deployment nel launcher.
