# PORT_PLAN — steering DS4 + transient steering da fork v0.1.38 su upstream v0.1.45

Tappa 1: sola analisi. Nessuna modifica al codice.

- **Destinazione**: `Strata-upstream-steering`, ramo `port/steering-v0.1.45`, HEAD `477ac13c` = tag `v0.1.45`.
- **Sorgente (read-only)**: `Strata-DualGPU-Pascal`, commit `afd832d85a933b917f9f5db1e38bbacb9165fa0f`
  ("feat(steering): implement DS4 directional steering and transient steering mechanism"), autore dino lupo,
  2026-10-06. Il fork parte da `v0.1.38` (commit `09817da`) + 16 commit dual-Pascal; tra v0.1.38 e v0.1.45
  ci sono **1.810 commit** → tutti i numeri di riga delle patch sono spostati.
- Patch esaminate: `port-steering/afd832d-cvec.cu.patch` (4 hunks), `afd832d-generate.cpp.patch` (12 hunks),
  `afd832d-server.py.patch` (4 hunks). Verificate con `git apply --check` hunk per hunk contro l'albero v0.1.45.
- File del commit `afd832d` (9 file, +839/−39) non contenuti nelle tre patch: `STEERING_LOG.md`,
  `data/experimental-speed-projection/Qwen3.8-Flash-Next-steering.f32` + `.json`,
  `tools/test_build_steering_vector.py`, `tools/test_steering_e2e.py`, `tools/test_steering_server.py`.
  Vanno presi come file nuovi (vedi §5). I tool `tools/ds4_to_cvec.py`, `tools/build_steering_vector.py`,
  `tools/steering_ab.py`, `tools/test_steering_e2e_runner.py` e `dir-steering/` sono nel commit precedente
  `9d10ad08` (fork), assenti in v0.1.45.

**Bilancio hunks**: 20 totali → **12 "si applica uguale"**, **8 "da adattare"**, **0 "non serve"**.
(Revisione: G4 passa da NON SERVE a UGUALE. Il kernel applica `dir` **due volte**, il segno della scala si
perde in v0.1.45: §4.4.)

Legenda stati: **UGUALE** = hunk applicato identico da `git apply --check`; **ADATTARE** = contenuto valido,
contesto/posizione da riscrivere a mano; **GIÀ PRESENTE** = la v0.1.45 fa già la stessa cosa; **NON SERVE** =
modulo ridondante. **Dopo la revisione di §4.4 la categoria NON SERVE è vuota**: prima contava 1 hunk nello
schema CUDA (ora 0) e il server Python era già vuoto (4 UGUALE su 4).

Convenzione delle colonne: "Fork (riga)" è la riga del file nel fork **dopo** il patch (il numero `+N` degli
header `@@ -a,b +N,m @@` dei file `.patch`); "v0.1.45" è la riga dell'albero `port/steering-v0.1.45`.

---

## 1. Tabella delle modifiche

### 1.1 `src/program/generate.cpp` (12 hunks)

| # | Cosa fa | Fork (riga) | v0.1.45 (file:riga) | Stato | Note sul conflitto |
|---|---|---|---|---|---|
| G1 | Campo `int cvec_tokens = 0` nella struct `Options` | `generate.cpp:551` | `src/program/generate.cpp:781-782` (dopo `cvec_single`) | UGUALE | `git apply` OK. Il commento `///` va allineato allo stile v0.1.45. |
| G2 | 4 righe d'aiuto `--cvec-tokens`, `--dir-steering-file`, `--dir-steering-ffn`, `--dir-steering-tokens` | `generate.cpp:646-649` | `src/program/generate.cpp:931-932` (dopo la riga `--cvec-dir`, prima di `--no-token-graph`) | UGUALE | `git apply` OK. |
| G3 | Caricamento del file `.f32` grezzo DS4 in `load_control_vectors` (GGUF provato prima, poi raw f32 full/headless) + `raw_ref` + controllo "dir-steering-ffn senza dir-steering-file" | `generate.cpp:1126-1230` | `src/program/generate.cpp:1814-1848` | ADATTARE | La firma v0.1.45 ha **`uint64_t& digest`** in più (`generate.cpp:1814-1815`): la patch non la conosce. Il `try` v0.1.45 usa `catch (const std::exception& e) { err = e.what(); }` (`:1846`); il patch vuole `catch (...)` + fallback f32: va preservato il `err = e.what()` nel ramo GGUF, altrimenti si perde la diagnosi. Il digest (§2.5) resta calcolato su `dir`/`s`, quindi il percorso f32 è coperto automaticamente. |
| G4 | Firma del fattore di scala in modo project: `sign = dot(d, raw_ref) < 0 ? -1 : 1`, `s = sign*nrm`, `dir = d/(sign*nrm)` | `generate.cpp:1242-1262` | `src/program/generate.cpp:1863-1872` | UGUALE | Applicato meccanicamente (offset) e **da portare** (`afd832d`: *preserve signed scale factor*). Il kernel usa `dir` **due volte** (`cvec.cu:66-70` nel `dot`, `:85-92` nell'aggiornamento), quindi con `s = nrm` sempre positivo (`:1866-1869`) il segno si quadra e sparisce: `--dir-steering-ffn -1` equivale a `+1` (§4.4). Dipende da G3, che fornisce `raw_ref`. Digest: con scala `1.0` `sign = +1` e il digest è identico alla v0.1.45 (profilo `experimental-speed-projection` invariato, §2.5); con scala negativa il digest cambia → le sessioni `.strata-sess` salvate con quella scala diventano invalide. Guardrail: `test_steering_e2e.py` Test 4. |
| G5 | Parsing `--cvec-tokens` / `--dir-steering-tokens` / `--dir-steering-file` / `--dir-steering-ffn` (file/scale in `cvec_files`, `cvec_mode = 0`) | `generate.cpp:1596-1615` | `src/program/generate.cpp:2573-2577` (blocco `--cvec-dir`, prima di `--no-spec-split` `:2578`) | UGUALE | `git apply` OK. In v0.1.45 `--cvec-mode` ha il alias `add` (`:2571`), da citare nell'help. |
| G6 | `int req_cvec_tokens = -1;` per richiesta (dopo `req_cvec`) | `generate.cpp:6133` | `src/program/generate.cpp:11636` (dopo `int req_cvec = 1;`) | ADATTARE | Il contesto v0.1.45 ha un commento diverso (`// ckpt=0: ...`, `:11637`) → il patch fallisce. Inserire dopo `:11636`. |
| G7 | Chiavi `cvec_tokens=` / `dir_steering_tokens=` nel parsing della riga di sampling | `generate.cpp:6150` | `src/program/generate.cpp:11671` (dopo `key == "cvec"`) | ADATTARE | In v0.1.45 la riga `cvec` è `:11671`, seguita da `ckpt` (`:11672`), `pin` (`:11673`), `temperature` (`:11674`): il patch ha contesto `temperature` e va riordinato dopo `cvec`. |
| G8 | `const int eff_cvec_tokens = req_cvec_tokens >= 0 ? req_cvec_tokens : o.cvec_tokens;` | `generate.cpp:6305` | `src/program/generate.cpp:11853` (dopo `const bool want_cvec = ...`) | UGUALE | `git apply` OK. `cvec_set_enabled(want_cvec)` è a `:12256` su un ramo separato. |
| G9 | Disarmo del transient nel **decoding pipelined** (finestre `K` verificate, output `outp`) | `generate.cpp:7434` | `src/program/generate.cpp:14056` (dopo `++produced_n;` di `:14056`) | UGUALE | `git apply` OK. Il ciclo è in `generate.cpp:14049-14064`. |
| G10 | Disarmo nel **decoding non pipelined** del server (output `outv`) | `generate.cpp:7697` | `src/program/generate.cpp:14396` (dopo `++produced_n;`) | ADATTARE | La v0.1.45 ha righe in più nel contesto: `if (sfx_on) sfx.append(...)` (`:14397`) e `extra_sources_append(...)` (`:14398`); il patch ha solo `if (o.suffix_draft > 0)`. Il corpo del disarmo è identico. |
| G11 | Disarmo nel ciclo CLI `sampled` (non speculativo), su `produced` | `generate.cpp:8341` | `src/program/generate.cpp:15238` (`if (pos >= n_prompt - 1) produced.push_back(next);`) | UGUALE | `git apply` OK. |
| G12 | Disarmo nel ciclo CLI speculativo (accettazione `outv`) | `generate.cpp:8635` | `src/program/generate.cpp:15687` (`produced.push_back(outv[i]);`) | ADATTARE | Contesto v0.1.45: `if (o.lookup_chain > 0) extra_sources_append(...)` (`:15689`) e la riga `follow_differ`/`follow_ok` (`:15690`) — assente nel fork. Corpo identico. |

Nota: i 4 punti di emissione dei token sono **solo questi** (`grep "++produced_n\|produced.push_back"`):
`14056`, `14396` (server) e `15238`, `15687` (CLI) — il patch ne copre esattamente 4, nessuna sede mancata.

### 1.2 `serve/server.py` (4 hunks)

| # | Cosa fa | Fork (riga) | v0.1.45 (file:riga) | Stato | Note sul conflitto |
|---|---|---|---|---|---|
| S1 | `projection_key` restituisce anche `cvec_tokens=N` (chiave `cvec_tokens` o `dir_steering_tokens`) | `server.py:557-565` | `serve/server.py:1232` (`return keys + StrataEngine.projection_key(sampling) + ...`) e `serve/server.py:1235-1239` | ADATTARE | Il contesto finale del patch è `def generate(...)`, in v0.1.45 segue `def _send(s, code, payload)` (`:1240`) → hunk rifiutato. La forma da scrivere: `keys = ""` + `keys += f" cvec={int(on)}"` + `keys += f" cvec_tokens={toks}"`, `return keys` — coerente con la funzione che in v0.1.45 costruisce una stringa di chiave (`:1232`). |
| S2 | `SHARED_KEYS` con `cvec_tokens`, `dir_steering_tokens` | `server.py:2966` | `serve/server.py:6514-6515` (tuple su due righe, già con `reasoning_budget_tokens` e `experimental_speed_projection`) | ADATTARE | In v0.1.45 la tuple è spezzata su due righe: aggiungere le due chiavi in coda a `:6515`. |
| S3 | Validazione in `clean_shared_defaults`: intero ≥ 0 per `cvec_tokens`/`dir_steering_tokens` | `server.py:2991-2997` | `serve/server.py:6518` (`def clean_shared_defaults`), ramo `seed` `:6544-6548`, ramo `reasoning_budget_tokens` `:6547-6550` (raise), ramo `experimental_speed_projection` `:6552` | ADATTARE | Il patch si aggancia al ramo `seed` del fork; la v0.1.45 ha un ramo in più (`reasoning_budget_tokens`) prima di `experimental_speed_projection`. Inserire il nuovo `elif` subito prima di `:6552`. La validazione di `reasoning_budget_tokens` è un buon modello (interi, ≥ 1; qui serve ≥ 0). |
| S4 | Validazione in `sampling_defaults_from_config` (config `sampling.cvec_tokens`) | `server.py:3051-3058` | `serve/server.py:6629` (`def sampling_defaults_from_config`), ramo `seed` `:6682-6685`, `experimental_speed_projection` `:6687` | UGUALE | `git apply` OK (contesto identico). Nota: v0.1.45 ha in più il ramo `reasoning_budget_tokens` (`:6689`) più avanti nell'elenco, non in conflitto. |

Chiavi `cvec_tokens`/`dir_steering_tokens` sono **nuove in v0.1.45** (`grep cvec_tokens` in `serve/server.py`: zero
occorrenze). Vanno aggiunte in tutti e 4 i punti: `projection_key` (chiave engine), `SHARED_KEYS`, le due
validazioni. Nessuna sede quinta.

### 1.3 `src/kernels/cuda/cvec.cu` (4 hunks)

| # | Cosa fa | Fork (riga) | v0.1.45 (file:riga) | Stato | Note sul conflitto |
|---|---|---|---|---|---|
| C1 | `DevTables` guadagna `volatile int* on_host` | `cvec.cu:15` | `src/kernels/cuda/cvec.cu:18` | UGUALE | offset 0. |
| C2 | `upload_here`: flag allocato con `cudaHostAlloc(Mapped|Portable)` + `cudaHostGetDevicePointer`; fallback `cudaMalloc` se il mapped non è disponibile | `cvec.cu:26-41` | `src/kernels/cuda/cvec.cu:26-38` | UGUALE | offset 0. L'ordine va rispettato: `t = DevTables{}` nell'errore **azzera anche `on_host`** (il patch lo fa). |
| C3 | Rilascio: `cudaFreeHost(on_host)` se mapped, altrimenti `cudaFree(on)` | `cvec.cu:126-128` (in `cvec_upload`) | `src/kernels/cuda/cvec.cu:110-113` (offset −16) | UGUALE | offset −16. `cudaDeviceSynchronize()` a `:110` resta (è al rinnovo del vettore, non nel disarmo). |
| C4 | `cvec_set_enabled`: scrittura `*on_host = v` su tutti i device; ramo di riserva `cudaMemcpyAsync` senza `cudaDeviceSynchronize()` | `cvec.cu:153-168` | `src/kernels/cuda/cvec.cu:137-150` (offset −17) | UGUALE | offset −17. Elimina il `cudaDeviceSynchronize()` (`:145`) che è la causa del deadlock. |

`include/strata/kernels/cvec.hpp` non cambia: `cvec_set_enabled(bool)` (`cvec.hpp:52`) e `cvec_enabled()`
(`:53`) hanno già la firma giusta; nessuna dichiarazione da aggiungere.

---

## 2. Cosa c'è già in v0.1.45 (infrastruttura control vector)

### 2.1 Kernel e tabelle — `src/kernels/cuda/cvec.cu`
- `Cvec g_cvec` (`:15`), `bool g_on_host` (`:16`), `constexpr int kDevices = 64` (`:17`),
  `struct DevTables { float* dir; float* s; int* on; }` (`:18`), `DevTables g_dev[64]` (`:19`),
  `std::vector<float> g_dir_host, g_s_host` (`:20`) — **le tabelle sono per-device** (layer split).
- `upload_here(err)` (`:26`): alloca `dir`, `s` e **il flag `on` in VRAM con `cudaMalloc` + `cudaMemcpy`**
  (`:29`, `:32`). È il punto che il patch converte in host-mapped.
- `cvec_kernel` (`:46`): un blocco per `(stream, token)`; legge `const int* on` e decide
  `steer = *on != 0 && s != 0` (`:53`); `mode == 0` calcola `dot = h·v` con reduce di warp (`:66-85`) e
  aggiorna `x -= dot*v`, `mode == 1` aggiunge `v` (`:91`). Il **flag è letto dentro il kernel**, quindi un
  valore host-mapped è visibile ai kernel lanciati dopo la scrittura.
- `cvec_upload(...)` (`:101-131`): libera le tabelle su tutti i device, carica le tabelle sul device corrente,
  pone `g_on_host = true` (`:120`), copia `dir/s/on` dentro `g_cvec` (`:123-130`) e marca `steered[l]` (`:130`).
- `cvec_replicate(err)` (`:133`): porta le tabelle sul device corrente (necessario con layer split).
- `cvec_set_enabled(on)` (`:137-151`): **`cudaDeviceSynchronize()` su ogni device con tabelle** (`:145`) poi
  `cudaMemcpy` del flag. `cvec_enabled()` (`:152`) legge `g_cvec.loaded() && g_on_host`: `g_on_host` è una
  **globale normale del processo** (`cvec.cu:16 bool g_on_host = false;`), **non thread-local** → lo stato è
  condiviso da tutti i thread del processo.
- `cvec_tables(&dir, &s, &on)` (`:154`): esporre le tabelle del device corrente — **solo v0.1.45** (serve al
  percorso prompt fusso, §2.4). Nel fork v0.1.38 non esiste.
- `cvec_apply(R, layer, T, r_ld, bo, bo_ld, inj, inj_ld, write, stream)` (`:162-171`): lancia `cvec_kernel`
  con grid `(hc, T)`, usando le tabelle del device corrente.

### 2.2 Applicazione nel decoding — `src/core/layer.cpp`
- `#include "strata/kernels/cvec.hpp"` (`:30`).
- `bool pending_ffn = fused && layer > 0 && !cvec().covers(layer - 1)` (`:1402`): il gate FFN dipende dal
  vettore.
- `const bool steer = cvec().covers(layer)` (`:1537`) → `cvec_apply(...)` (`:1541`, `:1544`, `:1547`):
  l'applicazione avviene **dopo la scrittura del layer**, su `bb.R`, per `T = 1`.

### 2.3 Opzioni CLI — `src/program/generate.cpp`
- Campi `Options`: `cvec_files` (`:778`), `cvec_first/last` (`:779`), `cvec_mode` (`:780`), `cvec_single` (`:781`).
- Help: `--control-vector-scaled FILE:SCALE[,...]` (`:927`), `--control-vector-layer-range A B` (`:929`),
  `--cvec-mode add|project` (`:930`), `--cvec-dir per-layer|single:L` (`:931`).
- Parsing: `--control-vector-scaled` (`:2548`), `--control-vector-layer-range` (`:2563`), `--cvec-mode` (`:2567`),
  `--cvec-dir` (`:2573`). Il profilo `experimental-speed-projection` passa
  `--control-vector-scaled FILE:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer`
  (commento `:776-777`).
- `load_control_vectors(o, g, summary, digest, err)` (`:1814-1898`): somma i tensori `direction.<l>` dei GGUF
  (`:1820-1845`), applica il range `first`/`last` e `single` (`:1849-1857`), in project mode calcola
  `nrm = ‖d‖`, `s[l] = nrm`, `dir[l] = d/nrm` (`:1863-1872`), poi `cvec_upload(...)` (`:1879`).
- Attivazione per richiesta: `req_cvec` (`:11636`), chiave `cvec` (`:11671`),
  `const bool want_cvec = cvec().loaded() ? req_cvec != 0 : true` (`:11853`),
  `cvec_cached` (`:11812`, `:11918`, `:11932`, `:11965`, `:12012`), e **`cvec_set_enabled(want_cvec)`
  chiamato a ogni richiesta** (`:12256`).

### 2.4 Percorso prompt (prefill) con vettore fusso
- `strata/kernels/prefill/fused_gr.cu:75` `gr_write_cvec_norm_rs_kernel` (kernel che fonde
  `gr_write` + `cvec_apply` + `gr_norm_rs`); helper `fused_gr` con `cvec_dir`/`cvec_s`/`cvec_on`.
- `src/prefill/prefill.cpp:444` (commento), `cvec_fuse()` (`:461`), `steered` (`:4510`),
  `cv_fused = steered && cvec_fuse() && ... && cvec_tables(&cv_dir, &cv_s, &cv_on)` (`:4513-4515`),
  lanci del kernel fusso (`:4537`, `:4556`) e il ramo non fusso `cvec_apply` (`:4535`, `:4568`).
  ⇒ `cvec_on` è letta anche qui: un disarmo host-mapped agisce sul prefill dei token successivi, e il flag
  viene letto da due famiglie di kernel.

### 2.5 Digest delle tabelle (legato alle sessioni)
`generate.cpp:1881-1892`: `SessionIdentityBuilder("CVEC")` con `mode`, `first`, `last`, `single`, `dir`, `scale`
→ `digest`; le sessioni `.strata-sess` sono vincolate al digest. Conseguenze:
- il percorso `.f32` del patch è coperto, perché il digest si calcola su `dir`/`s` già normalizzati;
- il segno di G4 entra nel digest solo quando `sign = -1`, cioè con scala negativa. Con scala `1.0` — il profilo
  `experimental-speed-projection` — `dir`/`s` sono identici alla v0.1.45 e le sessioni restano valide; con scala
  negativa il digest cambia e le sessioni `.strata-sess` salvate con quella scala diventano invalide;
- il default di `--dir-steering-file` pone `cvec_mode = 0` ⇒ il digest registra `mode=0`.

### 2.6 Protocollo server — `serve/server.py`
- `projection_key(sampling)` (`:1235-1239`): `cvec=0|1` da `experimental_speed_projection`;
  entra nella chiave engine in `keys` (`:1232`).
- `SHARED_KEYS` (`:6514-6515`): `reasoning_effort`, `reasoning_budget_tokens`, `temperature`, `top_p`,
  `top_k`, `seed`, `max_tokens`, `experimental_speed_projection`.
- `clean_shared_defaults` (`:6518`), rami `seed` (`:6544`), `reasoning_budget_tokens` (`:6547`),
  `experimental_speed_projection` (`:6552`); `sampling_defaults_from_config` (`:6629`) con gli stessi rami
  (`:6682`, `:6687`).
- `tools/cvec_parity.py` (`:1-104`): confronta `cvec=0|1` nel server. La v0.1.45 **non ha test di steering**;
  i test del fork (`tools/test_steering_e2e.py`, `test_steering_server.py`, `test_build_steering_vector.py`,
  `test_steering_e2e_runner.py`) sono assenti.

### 2.7 Port SYCL e Metal
- SYCL: `sycl/src/kernels/cuda/cvec.dp.cpp` (speculare a `src/kernels/cuda/cvec.cu`),
  `sycl/src/kernels/cvec_parity.cpp` (`cvec_set_enabled`/`cvec_enabled`/`cvec_upload`/`cvec_tables` `:76-92`),
  `sycl/src/core/layer.cpp` (`cvec_apply`, `covers` `:1039`, `:1169-1174`), `sycl/src/program/generate.cpp`
  (`req_cvec` `:7133`, `cvec` `:7160`, `want_cvec` `:7493`), `sycl/src/prefill/prefill.cpp`
  (`cvec_fuse` `:69`, `cvec_apply` `:1084`), `sycl/CMakeLists.txt` (`:17`).
- Metal: `engine/missing.cpp` (stub di `cvec_apply`).
- ⇒ **Fuori perimetro**: non abbiamo hardware SYCL/Metal da provare, quindi la replica del flag host-mapped in
  `sycl/src/kernels/cuda/cvec.dp.cpp` e `sycl/src/kernels/cvec_parity.cpp` (dove oggi `cvec_set_enabled` scrive
  `g_on_host` locale `:76-77`) **non viene fatta** in questo porting (§5).
  **Rischio documentato, non testato qui:** se il flag cambia semantica solo su CUDA, la parità SYCL/CUDA
  (`tools/cvec_parity.py`, che legge il flag lato host) diverge. Da riprendere quando si ha una scheda Intel Arc.

---

## 3. Dipendenze dal fork (codice che in v0.1.45 non esiste)

| Dipendenza del codice `afd832d` | Dove nel fork | Stato in v0.1.45 | Come la risolvo |
|---|---|---|---|
| `STRATA_XSTAGE` (hand-off zero-copy fra i due stadi di verifica Pascal, `cudaHostAlloc` dell'array `xflag`, `stage_ver(st).set_no_ple_late(true)`) | nel fork: `src/program/generate.cpp:5061-5074` (env), `:4738` (`xflag`), `include/strata/core/verify.hpp:121` (`set_no_ple_late`), `docs/DUAL_GPU.md:226` | **Assente** in v0.1.45 (`grep STRATA_XSTAGE` in `src/` e `serve/`: zero occorrenze) | Nessuna dipendenza reale: il disarmo del patch non legge `xflag`. È **l'unica opzione del fork assente in v0.1.45** (§5): da tenere fuori dal porting, serve solo per il porting dual-GPU (tappa separata). |
| Pipeline a finestre con `produced_n` nei due cicli di decoding del server | fork `:7434`, `:7697` | Presente con la stessa struttura: `produced_n` (`generate.cpp:13503`, `14056`, `14396`) | Nessuna risoluzione: i punti d'inserimento G9/G10 esistono, cambia solo il contesto circostante. |
| `--dir-steering-file` / `--dir-steering-ffn` / `--dir-steering-tokens` / `--cvec-tokens` | fork `:1596-1615` | Assenti in v0.1.45 (`grep --dir-steering`: zero) | Aggiungere i campi (`Options`, G1), l'help (G2), il parsing (G5). Sono le sole sedi. |
| Caricatore raw `.f32` con `raw_ref` | fork `:1126-1160` | Assente: `load_control_vectors` accetta solo GGUF (`generate.cpp:1820-1845`) | G3, da adattare alla firma con `digest` (`:1814`) e al `catch` v0.1.45 (`:1846`). |
| Firma dello scala in project mode | fork `:1242-1262` | v0.1.45 normalizza con `nrm` positivo (`:1866-1869`) | G4 **serve** (§4.4): `dir` entra **due volte** nel kernel, il segno della scala si quadra e si perde; da portare con G3. Digest invariato con scala `1.0`. |
| `extra_sources_append`, `follow_differ`/`follow_ok`, `lookup_chain`, `sfx_on`, `reasoning_budget_tokens` | assenti nel fork | Presenti solo in v0.1.45 (`generate.cpp:14398`, `15689-15690`; `server.py:6514`, `6547`) | Da preservare quando riscrivo G10/G12/S2/S3. |
| File dati `Qwen3.8-Flash-Next-steering.f32` + `.json` | `data/experimental-speed-projection/` nel commit `afd832d` | Assenti in v0.1.45 | Copiare i due file in `Strata-upstream-steering/data/experimental-speed-projection/` (tappa 3). |
| Tool di costruzione/conversione/AB | `tools/ds4_to_cvec.py`, `tools/build_steering_vector.py`, `tools/steering_ab.py`, `tools/test_steering_e2e_runner.py` (commit `9d10ad08`) | Assenti in v0.1.45 | Copiare come file nuovi (`tools/build_steering_vector.py` è citato dal test e serve per rigenerare il vettore). |
| Test di steering | `tools/test_steering_e2e.py` (usa `--dir-steering-file`, `--dir-steering-ffn`, `--dir-steering-tokens`, `--control-vector-layer-range`), `tools/test_steering_server.py`, `tools/test_build_steering_vector.py` | Assenti in v0.1.45 | Copiare in tappa 2 (passo 7), **prima** di eseguire i test. Da adattare solo i percorsi: `ENGINE = ROOT/"build-vector-steering"/"strata"` → la build di questa worktree, `PACK` (`Strata-data`), `NATIVE_SHARD`, i due vettori. **Nessuna sostituzione di flag**: `--resident-experts`, `--resident-budget-gib`, `--vram-reserve-mib`, `--expert-profile`, `--mtp`, `--prefill`, `--spec`, `--pipeline-windows`, `STRATA_PIN_GUARD=2` esistono già in v0.1.45; `--stage` non è un flag della release. Nessun test legge `STRATA_XSTAGE`. |
| Architettura del modello `qwen4exp` | `dir-steering/build_ds4_vector.py` (`:24` "Qwen3.8 Flash Next, 48 layers, 2560 hidden"), `STEERING_STUDY.md` §3 | v0.1.45 ha il modello MoE (`mtp`, `Qwen3.8-Flash-Next`, `data/experimental-speed-projection/`) | **Ipotesi**: da verificare che la geometria del modello v0.1.45 sia 48×2560 come il `.f32` del vettore: `load_control_vectors` richiede `elements == n_embd` (`:1829`) e il raw `.f32` richiede `(L-1)*N*4` o `L*N*4` byte. |

Il codice `afd832d` **non usa** `set_commit_async`, `pl_commit_async`, `remote_expert_opt` né API di stream
condizionali: le sue uniche dipendenze dal fork sono (a) le sedi di inserimento nei cicli di decoding
(`produced_n`/`produced`), (b) le opzioni CLI `--dir-steering-*`, (c) il caricatore `.f32`, (d) i file dati e i
tool. La pipeline dual-GPU e `STRATA_XSTAGE` **non sono necessari** per il porting dello steering. Verificato:
`--pipeline-windows` (55 occorrenze), `--trim-stage-weights`, `--resident-experts` (11), `--resident-budget-gib`,
`--vram-reserve-mib`, `--expert-profile`, `--spec`, `STRATA_PIN_GUARD` sono nativi di v0.1.45;
**solo `STRATA_XSTAGE` è assente**.

---

## 4. Il punto delicato: disarmo del transient steering senza `cudaDeviceSynchronize`

### 4.1 Il problema (Fase 6 di `STEERING_LOG.md:119-126`)
Nel fork, con `--pipeline-windows 2`, `cvec_set_enabled(false)` eseguiva `cudaDeviceSynchronize()` su GPU 0 e
GPU 1 (**`v0.1.45 cvec.cu:145` è esattamente questo codice**) mentre i due stadi di verifica si coordinano con
le hand-off flags: il thread host si è bloccato sul driver → deadlock dopo 5 token. La causa non è il flag in
sé, è la **sincronizzazione completa del device chiamata in mezzo a una coda di graph in volo**.

### 4.2 La soluzione del fork, e perché è idiomatica in v0.1.45
Pinned host memory zero-copy (`cudaHostAlloc(..., cudaHostAllocMapped | cudaHostAllocPortable)` +
`cudaHostGetDevicePointer`), con il flag `on` che diventa un indirizzo device-mapped e un alias
`volatile int* on_host` sul lato host. La scrittura `*on_host = v` è una store CPU: nessun `cudaDeviceSynchronize`,
nessun lock del driver, nessuna ricostruzione dei graph. La v0.1.45 usa già questo schema:
`remote_expert_opt.cu:58-60`, `verify.cpp:379-384` (flags `ready`/`done` fra tier di peer),
`layer.cpp:523-534` (copia KV su host pinned), `weights.cpp:257`. Il `commit` del decoding usa
`h_commit_` pinned + `std::atomic_thread_fence(std::memory_order_seq_cst)` (`src/core/verify.cpp:3211`,
`set_commit_async` `:541`, `pl_commit_async` `:4174`) — nessun `cudaDeviceSynchronize` nel percorso normale.
⇒ il disarmo transient entra nella famiglia di pattern già usata, non introduce un meccanismo nuovo.

### 4.3 Design per v0.1.45
1. `DevTables` + `on_host` (`cvec.cu:18`), `upload_here` con mapped + fallback `cudaMalloc` (`cvec.cu:26`),
   rilascio corretto (`cvec.cu:110-113`), `cvec_set_enabled` con store host (`cvec.cu:137-150`) — i 4 hunks
   C1-C4 si applicano identici.
2. Il flag è letto da `cvec_kernel` (`cvec.cu:53`) e dal percorso fusso del prefill
   (`prefill.cpp:4513-4515`, `fused_gr.cu:75`): entrambe lo leggono **all'interno del kernel**, quindi una
   scrittura host è presa dai kernel lanciati dopo, inclusi quelli dei graph (`grid (hc, T)`, `cvec_apply`
   `:164`). Non serve toccare la cattura dei graph.
3. Sede del disarmo: dopo `++produced_n` a `generate.cpp:14056` (pipelined) e `:14396` (non pipelined), e dopo
   `produced.push_back` a `:15238` e `:15687` (CLI). Il disarmo scrive il flag, non sincronizza: il decoding
   pipelined continua a_launchare le finestre successive.
4. Riattivazione: `cvec_set_enabled(want_cvec)` è chiamato a **ogni** richiesta (`generate.cpp:12256`),
   quindi un disarmo a metà richiesta non contamina la richiesta seguente; `cvec_cached` (`:11812`) governa la
   cache del prompt, non il flag. Da conservare così com'è.
5. Semantica da decidere e documentare: il disarmo è **di processo**, non di richiesta — `g_on_host` è globale
   nel processo (`cvec.cu:16`). Se due finestre sono in volo su GPU diverse, il flagmapped è condiviso
   (`Portable`), quindi il disarmo vale per tutte. Va scritto nell'help che `--dir-steering-tokens N` disarma
   per tutto il processo fino alla prossima richiesta.
6. Ordine dei token: `produced_n` conta i token emessi dal server, il flag agisce sui kernel lanciati dopo la
   scrittura; la finestra di finestrelle già in volo può applicare il vettore a qualche token oltre N. Da
   misurare con il test E2E (**Ipotesi:** l'off-di-N è di una frazione di finestra in modalità pipelined; nel
   fork Fase 6 il disarmo è stato osservato "dopo 5 token" con `--dir-steering-tokens 5`).
7. SYCL/Metal: **fuori perimetro** — non abbiamo hardware da provare, quindi il flag mapped **non** viene
   replicato in `sycl/src/kernels/cuda/cvec.dp.cpp` né in `cvec_parity.cpp` (`:76-77`) in questo porting (§2.7,
   §5). Da riprendere con una scheda Intel Arc; il rischio è la divergenza di `tools/cvec_parity.py`.
8. Se il mapped fallisce (WSL2, driver vecchi), il fallback `cudaMalloc` + `cudaMemcpyAsync` (`cvec.cu` ramo
   `on` del patch) mantiene il comportamento v0.1.45 senza `cudaDeviceSynchronize` — la degradazione è
   silenziosa: `cvec.cu:145` va eliminato in **entrambi** i rami.

### 4.4 Perché G4 (segni) serve
Il kernel (`cvec.cu:66-92`) applica `dir` **due volte**: `dot = Σ x_j·dir_j` (`:66-70`), poi `dot *= s` (`:84`),
poi `x_j -= dot·dir_j` (`:85-92`, `fmaf(-dot, v[d], xv)` con `v = dir`). L'aggiornamento è
`x -= s·(x·dir)·dir`.

Fork: `s' = sign·nrm`, `dir' = d/(sign·nrm)`. v0.1.45: `s = nrm` (sempre positivo, `generate.cpp:1866-1869`),
`dir = d/nrm`. Con `d = scale·v`, `nrm = |scale|·‖v‖`, `sign = sign(scale)`, `v̂ = v/‖v‖`:

- **fork**: `x - s'·(x·dir')·dir' = x - sign·|scale|·‖v‖·(x·v̂)·v̂` → il segno resta
  (positivo = soppressione, negativo = amplificazione; è il fix di `afd832d`: *preserve signed scale factor*).
- **v0.1.45**: `dir = sign·v̂`, quindi
  `x - nrm·(x·sign·v̂)·(sign·v̂) = x - |scale|·‖v‖·sign²·(x·v̂)·v̂ = x - |scale|·‖v‖·(x·v̂)·v̂`.
  `sign² = 1` ⇒ **il segno della scala si cancella**: `--dir-steering-ffn -1` è identico a `+1`, perché `dir`
  entra nel `dot` **e** nell'aggiornamento.

⇒ G4 **va portato** (con G3, che dà `raw_ref`): il segno si sposta da `dir` a `s`, dove non si quadra. Il solo
prodotto `s·(x·dir)` è identico nei due modi — è il secondo `dir` a rendere reale la differenza.

Custodia del digest (§2.5): con scala `1.0` `sign = +1`, `dir`/`s` e il digest sono identici alla v0.1.45 (il
profilo `experimental-speed-projection` non cambia); con scala negativa il digest cambia.

Guardrail: `test_steering_e2e.py` Test 4 (*Negative scale … Signed-scale output effect was not demonstrated*)
fallisce con la v0.1.45 nuda ⇒ prova che G4 è necessario.

Eccezione da verificare: con **più file** sommati con scale di segno misto, `sign = dot(d, raw_ref)` decide il
segno della somma dominante; il segno del singolo file non è preservato (**Ipotesi:** non coperto dai test).

---

## 5. Piano delle tappe 2-3

Ordine minimo, file per file; ogni passo è verificabile senza GPU (build + test locale), e i passi 1-2
non cambiano il comportamento (`cvec_tokens = 0` = tutti i token, nessuna opzione nuova = v0.1.45 identica).

**Tappa 2 — CUDA, opzioni, server (parity + transient)**
1. `src/kernels/cuda/cvec.cu`: applicare C1→C2→C3→C4 nell'ordine (`:18`, `:26`, `:110-113`, `:137-150`),
   verificare che il ramo `on` di riserva non contenga `cudaDeviceSynchronize`.
2. `src/program/generate.cpp`: G1 (`:782`) + G2 (`:932`) + G5 (`:2578`) = campo, help, parsing
   `--cvec-tokens`/`--dir-steering-*`. Nessun cambiamento di comportamento con `cvec_tokens = 0`.
   Verifica: `--help` e `--dir-steering-file` con file GGUF (`--dir-steering-file` → `cvec_mode = 0`).
3. `src/program/generate.cpp`: G3 (`:1814-1848`) caricatore `.f32` con `raw_ref`, con la firma `digest`,
   il `catch` v0.1.45 e l'errore "dir-steering-ffn senza dir-steering-file"; **e G4** (`sign` da `dot_ref`,
   `s = sign*nrm`, `dir = d/(sign*nrm)`, §4.4).
   Verifica: parità byte-per-byte del digest tra GGUF e `.f32` generato da `ds4_to_cvec.py`
   (`cvec_parity.py:76-86` è il modello di questo confronto) e INFO cvec identico con scala `1.0`;
   con scala `-1.0` il digest deve risultare **diverso** da quello di v0.1.45 (Test 4 del passo 8).
4. `src/program/generate.cpp`: G6 (`:11636`), G7 (`:11671`), G8 (`:11853`) = chiave di sampling
   `cvec_tokens=`/`dir_steering_tokens=` e `eff_cvec_tokens`.
5. `src/program/generate.cpp`: G9 (`:14056`), G10 (`:14396`) disarmo server; G11 (`:15238`), G12 (`:15687`)
   disarmo CLI. Preservare `extra_sources_append` (`:14398`, `:15689`) e `follow_differ` (`:15690`).
6. `serve/server.py`: S1 (`:1235-1239`), S2 (`:6515`), S3 (`:6552`), S4 (`:6686`).
7. Copiare i dati e i tool — **necessari per eseguire i test, quindi prima del passo 8**:
   `data/experimental-speed-projection/Qwen3.8-Flash-Next-steering.f32` + il `.json` del fork nella directory
   omonima di v0.1.45 (SHA-256: `29ee1a8989772791a044d849573757616fb55eb14eb1b750651b05113bf9a061` verificato con `sha256sum`; il file non
   va letto a schermo, solo copiato, §Regola dei dataset), il sidecar `Qwen3.8-Flash-Next.json` del fork (usato
   dai tool per il layer range), `tools/ds4_to_cvec.py`, `tools/build_steering_vector.py`,
   `tools/steering_ab.py`, `tools/test_steering_e2e_runner.py`, `tools/test_build_steering_vector.py` (che ha
   il `.f32` di prova), `tools/test_steering_e2e.py`, `tools/test_steering_server.py`.
   Da adattare **solo i percorsi**: `ENGINE = ROOT/"build-vector-steering"/"strata"` → la build di questa
   worktree, `PACK`/`NATIVE_SHARD` (`Strata-data`), i due vettori. **Nessuna sostituzione di flag**: i flag che
   i test richiedono (`--resident-experts`, `--resident-budget-gib`, `--vram-reserve-mib`, `--expert-profile`,
   `--mtp`, `--prefill`, `--spec`, `--pipeline-windows`, `STRATA_PIN_GUARD=2`) esistono già in v0.1.45; `--stage`
   non è un flag della release; nessun test legge `STRATA_XSTAGE` (§3).
8. Test: `python3 tools/test_steering_e2e.py` (GGUF, `.f32`, **scala negativa** = guardrail di G4/§4.4,
   `--dir-steering-tokens 5`) e `tools/test_steering_server.py` con `--pipeline-windows 2`;
   `tools/cvec_parity.py` come guardrail. Verifica di non regressione: `tests/` e i test del server di v0.1.45.

**Tappa 3 — documentazione**
9. `STEERING_LOG.md` e `dir-steering/STEERING_STUDY.md` nel worktree (documentazione dello steering);
   aggiungere le righe d'aiuto anche al profilo `experimental-speed-projection` se il profilo deve
   esporre `--dir-steering-tokens`.
10. Verifica finale: `--pipeline-windows 2` + `--dir-steering-tokens N` su GPU 0 sola (un device basta per il
    disarmo), poi dual-GPU se il porting dual-Pascal è già integrato.

**Fuori dal perimetro di questo porting**: **solo `STRATA_XSTAGE`** — è l'unica opzione del fork assente in
v0.1.45. `--pipeline-windows`, `--trim-stage-weights` e `--resident-experts` sono nativi della release (§3),
quindi non sono lavoro di porting. SYCL e Metal restano fuori perimetro perché **non abbiamo hardware da
provare** (§2.7, §4.3): la replica del flag mapped non si fa. Lo steering funziona anche con una sola GPU.

---

## 6. Non verificato

- **Non ho compilato nulla** (regola della tappa 1): i 4 hunks `cvec.cu` passano `git apply --check`, ma
  `cudaHostGetDevicePointer` su una `cudaMalloc` della stessa `cudaHostAlloc` in contesto di peer/device multipli
  non è stato provato su 2 GPU. **Ipotesi:** `cudaHostAllocPortable` è sufficiente con layer split.
- **Semantica del disarmo con più thread**: `g_on_host` è una **globale normale del processo** (`cvec.cu:16`),
  non thread-local: `cvec_enabled()` (`:152`) la legge da qualunque thread, quindi il disarmo è di processo e
  vale per tutti. Non ho verificato se `serve/server.py` gestisce richieste parallele su più thread.
  **Ipotesi:** v0.1.45 serve una richiesta alla volta (`generate.cpp:12251-12256` usa un unico `cvec_cached`),
  quindi lo stato globale di processo è sufficiente e la riattivazione a inizio richiesta (`:12256`) lo
  ristabilisce.
- **Off-di-N del disarmo in modalità pipelined**: quanti token oltre N vedono il vettore quando una finestra
  K è già in volo. Non misurato (richiede GPU). **Ipotesi:** al massimo una finestra di anticipo.
- **Equivalenza GGUF↔`.f32` sul modello v0.1.45**: il file `.f32` del fork è per il modello MoE
  `Swift-Qwen3.8-Flash-Next` (48×2560, `STEERING_STUDY.md` §3). La geometria del modello usato da v0.1.45
  non è stata confrontata con `g.n_layers`/`g.n_embd` di `load_control_vectors` (`:1816`). Da fare in tappa 2.
- **Semantica di `--dir-steering-ffn` senza `--dir-steering-file`**: il patch spinge `{"", sc}` in `cvec_files`
  e `load_control_vectors` lo rifiuta con un messaggio; l'ordine degli argomenti conta. Non verificato in CLI.
- **`reasoning_budget_tokens`**: v0.1.45 ha questa chiave in `SHARED_KEYS` (`:6514`) e nei rami di validazione
  (`:6547`, `:6689`); il modo esatto in cui interagisce con `max_tokens` non è stato letto.
- **Port SYCL/Metal — fuori perimetro, non implementato e non verificato**: non abbiamo hardware da provare,
  quindi il flag mapped non viene replicato in `sycl/src/kernels/cuda/cvec.dp.cpp` né in
  `cvec_parity.cpp:76-92` (`upload_here` SYCL andrebbe letto riga per riga). Rischio documentato (§2.7):
  divergenza di `tools/cvec_parity.py` se la semantica del flag cambia solo su CUDA. Tappa separata, con
  hardware Intel Arc disponibile.
- **Test di steering su v0.1.45**: assenti (`grep cvec tests/` → zero; `grep steering serve/` → zero). La
  parità del percorso prompt fusso (`prefill.cpp:4513`, `fused_gr.cu:75`) con il flag mapped non è coperta da
  nessun test.
- **`git apply` con offset**: gli offset riportati (`cvec.cu` −16/−17, `generate.cpp` h1 +232, h5 +648,
  h8 +594, h9 +498, h11 +897) sono l'effetto dei 1.810 commit; non è stata verificata la stabilità degli
  anchor su `main` (HEAD `01917e91`), solo su `v0.1.45`.
- **Numeri di riga delle sedi CLI/serve**: presi da `grep` sul file v0.1.45 (`11636`, `11671`, `11853`, `12256`,
  `14056`, `14396`, `15238`, `15687`); l'inserimento va fatto con contesti di ±3 righe, non con numeri fissi.
- **Contesto del patch `generate.cpp` h3**: l'errore `"--dir-steering-ffn specified without --dir-steering-file"`
  è nel patch, ma il `catch` v0.1.45 (`:1846`) usa `e.what()`; l'interazione dei due rami (GGUF valido ma
  architettura diversa, file vuoto) non è provata.

## Stage 2 log

- Step 0: SHA-256 .f32 corretto in §5 (29ee1a89… verificato con sha256sum) — commit `5a910a29`, build n/a
- Step 1: cvec.cu C1-C4 (git apply, offset 0; ramo di riserva senza cudaDeviceSynchronize) — commit `12154e97`, build OK
- Step 2: generate.cpp G1+G2+G5 (apply offset 231/285/985) — commit `c5efeed9`, build+help OK; dir-steering-file GGUF runtime check rinviato (GPU)
- Step 3: generate.cpp G3 (raw .f32 loader + raw_ref + dir-steering-ffn errore) adattato con e.what() preservato, G4 (signed scale) — commit `cb88a389`, build OK; esempio numerico G4: -1.0 amplifica, noG4 -1.0==+1.0; GGUF/f32 runtime check rinviato (GPU)
