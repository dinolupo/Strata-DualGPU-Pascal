# Studio: Directional Steering di ds4 (antirez) e l'implementazione Strata

Documento di studio per l'uso e la generazione di vettori direzionali su
Qwen3.8 Flash Next. Basato sull'analisi diretta del codice ds4
(`ds4.c`, `ds4_cuda.cu`, `dir-steering/tools/build_direction.py`) e del
codice Strata di questo repository (`src/kernels/cuda/cvec.cu`,
`src/program/generate.cpp`, `tools/build_steering_vector.py`), non solo dai README.
Revisione 2: corregge due errori della prima versione e colma le lacune
segnate nel controllo del 09/10.

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

1. Per ogni coppia di prompt `(good, bad)` lancia il modello con `-n 1`
   **dentro il chat template del profilo** con un system prompt
   (`--system`, default "You are a helpful assistant."; righe 88-89, 133).
   L'ultimo token del prompt è quindi quello in cui l'assistente sta per
   rispondere: il punto dove il comportamento si decide.
2. Abilita il dump della riga **`ffn_out`** (uscita FFN) dell'**ultimo token
   del prompt**, per ogni layer (`DS4_METAL_GRAPH_DUMP_*`).
3. `direction[layer] = normalize(mean(tutte good) − mean(tutte bad))`
   (media globale), oppure con `--pair-normalize`: normalizza la differenza
   di ogni coppia prima di mediare.
4. **Ortogonalizzazione di Gram-Schmidt** (default ON, `--no-orthogonalize`
   la disattiva): rimuove dal vettore la componente parallela alla media dei
   control (`bad`), rinormalizza (righe 186-198). Motivo: il vettore deve
   catturare solo la *differenza* target−control, non il contesto condiviso.
5. Scrive un JSON di metadati (`format: ds4-directional-steering-v1`, shape,
   component, provenienza) e il `.f32` piatto.

## 3. Semantica a runtime (corretta)

L'edit applicato a ogni layer, su ogni riga (token) del tensore:

```
y = y − scale · dir · dot(dir, y)
```

`dot(dir, y)` è la componente del token attuale lungo l'asse catturato.
Per componenta lungo quell'asse:

- `scale = 1`: **azzera** la componente → il modello diventa **neutro**
  sull'asse, non "sopprime il good" e non spinge verso il bad.
- `scale = 2`: la componente si **inverte** (valore negativo sull'asse).
- `scale < 0`: **amplifica la componente che il token ha già**, qualunque
  sia il suo segno: su un token già allineato al good la rafforza, su un
  token allineato al bad la rafforza in negativo.
- `scale = 0` o file assente: percorso normale.
- Default ds4: `--dir-steering-ffn 1` (col file), `--dir-steering-attn 0`.
- Kernel: `directional_steering_project_kernel` (`ds4_cuda.cu:12451`),
  dot con riduzione per riga, poi `x −= scale·dot·dir`.

## 4. Dove agisce su Qwen3.8 Flash Next (ds4.c:58058-58068)

- **FFN steering**: proietta il tensore **R** (flusso residuo) su **tutte le
  4 branche hyperconnection** (`rows = T × DS4_N_HC`) subito dopo il FFN di
  ogni layer, prima dell'espansione HC.
- **Attention steering** (ds4-only): proietta l'output attention (`blk`)
  prima dell'iniezione nel residuo. Default 0; il README ds4 lo definisce
  "fragile" e sperimentale.
- In **Strata** il kernel `cvec` applica la proiezione sul **folded write del
  layer** (il residuo, tutte le stream HC): è l'equivalente del lato FFN.
  **Il lato attention non è implementato** (dichiarato nel README di
  `dir-steering/README.md`).
- **Range di layer in Strata**: `--control-vector-layer-range A B`
  (default `1 .. L-1`) e `--cvec-dir single:L` (`generate.cpp:546-550,
  1230-1232`). Il vettore attuale è progettato per i **layer 4-44**:
  applicare su 1..47 include layer dove il vettore non è calibrato.

## 5. Over-saturazione e transient steering (ipotesi, non misurata in Strata)

Il README ds4 riporta osservazioni locali (greedy, prompt singoli):
ripetizione, deriva grammaticale, perdita di contenuto dopo ~20-40 token di
steering continuo; scale negativi forti (−2, −3) collassano in ripetizione
su alcuni prompt. **In Strata non è stato misurato nulla di sistematico**:
va trattata come ipotesi da qualificare con la suite di valutazione
(section 10). Il transient steering (`--dir-steering-tokens N`) è il
meccanismo di mitigazione; in Strata il disarmo avviene via flag in host
memory zero-copy (`cudaHostAlloc mapped`), senza `cudaDeviceSynchronize` né
contese sul driver. Verificato in Strata: disarmo esatto al token 5/12/20,
log `deactivated after N tokens`, nessuno stallo, reset pulito.

## 6. Il vettore attuale nel repository: è una PROIEZIONE DI RIFIUTO

`Qwen3.8-Flash-Next-steering.f32` /
`Qwen3.8-Flash-Next-experimental-speed-projection.gguf` derivano da
**coppie di prompt reali** (attivazioni Qwen3.8-Flash-Next), e il README
del pacchetto (`data/experimental-speed-projection/README.md`) lo dichiara:

- è una **refusal-direction projection**: con il vettore attivo il modello
  rifiuta 1 richiesta su 50 invece di 50 su 50 (test set dei suoi autori);
- i file pubblicati hanno "refusal" nel nome (le bytes sono le stesse);
- agisce sui **layer 4-44**, non su tutti i 48;
- rimuove un comportamento di sicurezza: la responsabilità di ciò che il
  modello scrive resta di chi lo usa;
- costa 0,2-0,4 % per token su testo normale, con lieve shift delle
  previsioni (misurato in `bench/results/2026-09-27-esp/`).

**Conseguenza per i test fatti finora**: tutti gli A/B e gli e2e (inclusi
quelli di oggi) hanno misurato la **rimozione del rifiuto**, non uno stile
neutro. I test di parità di formato e di meccanismo restano validi; la
loro interpretazione comportamentale va letta così.

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
  --out out/refusal-direction \
  --engine build-vector-steering/strata \
  --pack /home/dino/proj/Strata-data/packs/swift-iq2_xs \
  --ple-gguf /home/dino/proj/genAI/models/Qwen3.8-Flash-Next-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
```

**Nota sui nomi**: `harmless_alpaca` contro `harmful_behaviors` produce
**di nuovo una direzione di rifiuto** (stessa geometria del vettore
attuale), non un "concept steering". Per una direzione di stile usare
coppie stile-vs-stile (es. `dir-steering/examples/succinct.txt` contro
`verbose.txt`).

### Due lacune metodologiche nel builder Strata (da correggere)

1. **Nessun chat template, nessun system prompt**
   (`build_steering_vector.py:136`: tokenizza il testo grezzo). ds4
   tokenizza dentro il template con system prompt, così il token catturato
   è quello in cui l'assistente inizia a rispondere. Questo cambia la
   direzione **più** della differenza `ffn_out` vs residuo (section 8).
   Correzione facile: riusare `chat_prompt()` di `steering_ab.py`.
2. **Dump del tensore sbagliato**: `--dump-layers` scrive R dopo ogni layer
   mediato sulle 4 stream HC (`build_steering_vector.py:175-188`), non
   `ffn_out` (section 8).

## 8. `ffn_out` vs residuo: cosa succede senza la modifica

**ds4 cattura `ffn_out`** (uscita FFN, prima dell'iniezione HC).
**Strata `--dump-layers` cattura `R` dopo ogni layer**, mediato sulle 4
stream HC. Non sono lo stesso tensore.

Se NON si implementa il dump fedele di `ffn_out` in Strata:

1. **Coerenza interna**: cattura su R-post-layer e applicazione sul folded
   write (R) avvengono nello **stesso spazio**: un vettore generato da
   Strata è autocoerente e produce direzioni valide.
2. **Contaminazione della direzione**: R-post-layer contiene FFN **più**
   attention **più** il residuo accumulato: rapporto segnale/rumore
   inferiore per il solo comportamento FFN; scale da calibrare
   empiricamente, effetti collaterali più ampi.
3. **Non intercambiabilità con i vettori ds4**: i `.f32` costruiti da ds4
   (spazio `ffn_out`) applicati in Strata (spazio R) sono un'
   approssimazione — è ciò che accade oggi con il vettore refusal.
4. **Riproducibilità**: un vettore generato da Strata non è confrontabile
   con uno ds4 per le stesse coppie di prompt.

Se si implementa il dump `ffn_out` fedele: vettori semanticamente
identici a ds4 per le stesse coppie; intercambiabilità dei `.f32`.
Costo: estensione del grafo CUDA (zona critica: graph capture, pipeline
dual-GPU), da qualificare con la suite e2e.

## 9. Effetti specifici di Strata su MoE + MTP (da misurare)

- **Selezione degli esperti**: modificare il residuo cambia quali esperti
  vengono routerati → cambia la hit rate della cache VRAM e le letture dal
  disco, che oggi sono il collo di bottiglia.
- **Draft MTP non steered**: con steering attivo le bozze (generate dal
  predittore non steered) possono essere accettate meno spesso →
  generazione più lenta.
- Entrambi si misurano dalle righe `drafts accepted` / hit rate dei log
  engine: confrontare baseline vs steered sulla stessa generazione.
  Il costo dichiarato del vettore attuale (0,2-0,4 %/token) è del pacchetto
  originale, non misurato su questa build.

## 10. Metodo di valutazione con numeri (protocollo da implementare)

L'A/B attuale verifica solo che l'output "sia diverso". Serve:

1. **Set di valutazione separato** dai prompt usati per costruire il
   vettore (held-out), con prompt neutri + prompt target.
2. **Metrica del comportamento obiettivo**: tasso di rifiuto (50 richieste
   classify), o lunghezza/stile per direzioni di stile.
3. **Metrica dei danni collaterali su testo normale**: perplexity o
   divergenza KL delle distribuzioni di logits baseline vs steered, a
   scale `0, ±0.5, ±1, ±2` (il protocollo ds4: iniziare tra −1 e +2;
   ripetizione o perdita di contenuto = scala troppo forte).
4. **Velocità**: token/s, drafts accepted, hit rate cache, a ogni scala.

## 11. Cache delle conversazioni con lo steering

Il riuso della cache funziona solo con **matching steering mode**
(`docs/DETAILS.md:582`: "only exact token/image prefixes with matching
steering mode are reused"). Attivare o disattivare lo steering a metà di
una conversazione costa una **rilettura completa** (8-10 minuti a
80.000 token). Il transient steering aggrava il caso: il disarmo a N
token cambia la modalità a metà conversazione. Protocollo d'uso:
impostare la modalità steering all'inizio della conversazione e non
cambiarla; il per-request `cvec=0|1` va trattato come cambio di modalità.

## 12. Stato di qualifica attuale (branch `feat/vector-steering`)

- Unit: matematica vettori, round-trip `.f32`↔`.gguf` bitwise, 9 regression
  test CPU sull'harness.
- Kernel: `cvec_parity` (on/off, add, project, signed scale).
- E2E: parità bitwise `.f32`↔GGUF su output reali, transient 5/12, reset
  pulito.
- A/B comportamentale: ffn=−1 e ffn=+2 differiscono da baseline e tra loro
  su 3 prompt; contesto 131072 con disarmo al token 20 (7,4 s, 117 token),
  evento nel log, nessun `Xid`.
- **Aperto, con protocollo**: conversazione lunga reale (~58K prompt) →
  scale da scegliere (es. −1), transient N (es. 20), metriche della
  section 10 (rifiuto, KL, token/s, drafts accepted), costo cache della
  section 11.
