# AGENTS.md

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.
- **Directional Steering and Research Documentation**: Always keep Strata's documentation (`dir-steering/README.md`,
  `STEERING_LOG.md`, and related notes) updated and aligned with ongoing studies and experimental findings.
- **STRICT PROHIBITION ON PARSING DATASETS**: Agents must NEVER inspect, parse, display, read, or print the raw text
  contents of test/steering datasets (e.g., in `datasets/`, `mlabonne/`, or any parquet/txt prompt files). It is
  strictly forbidden for agents to view or output dataset text. All dataset processing must be executed strictly by
  standalone local scripts operating on file paths, reporting only progress, row counts, tensor shapes, and status.
- **DIVIETO ASSOLUTO DI AZIONI SILENTI E COMPILAZIONI NON AUTORIZZATE (ZERO-SURPRISE POLICY)**:
  - L'autorizzazione riguarda l'attività: una volta approvati build, test o inferenza, diagnosi, correzioni e ripetizioni pertinenti non richiedono nuove conferme. Dare un breve preavviso prima delle fasi pesanti; chiedere consenso solo per attività nuove o fuori dal perimetro approvato.
  - Correggere autonomamente errori recuperabili di directory di lavoro, percorso/interprete, lettura, patch, flag, build e test. Verificare la causa e ripetere la verifica dopo una correzione mirata. Non mascherare errori aumentando timeout o indebolendo le assertion.
  - Per OOM o stalli di test isolati, interrompere e ripulire soltanto i processi avviati dall'agente, verificare le risorse e diagnosticare/riprovare in sicurezza senza chiedere nuovamente. Non terminare processi dell'utente; documentare variazioni di budget, ordine GPU e configurazione nei confronti.
  - Riportare errore esatto, causa verificata o ancora ignota e intervento in italiano semplice. Limitare a tre tentativi mirati per lo stesso problema; poi dichiararlo bloccato e proseguire solo con attività indipendenti sicure. Chiedere indicazioni solo per decisioni dell'utente, ampliamenti del perimetro o rischi persistenti per il sistema.
  - Restano vietate modifiche a repository esterni (es. `pascal-frankenstein-llm`, `llama.cpp`) e modifiche non autorizzate a produzione/steering. Non sostituire il binario stabile e non eseguire commit, push o deployment impliciti.
  - Trasparenza assoluta e onestà: mai lasciare l'utente all'oscuro, mai lanciare processi lunghi in background senza preavviso, e mai prendersi il merito di azioni eseguite dall'utente (es. terminazione di task).
