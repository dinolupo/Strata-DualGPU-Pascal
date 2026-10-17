# Revisione di `STEERING_STUDY.md` (09/10/2026)

Revisione fatta sul codice di ds4 e di Strata in questo repository. Ogni punto ha la sua fonte.
Da applicare allo studio e da tenere presente nelle prossime analisi (vedi la checklist in `AGENTS.md`).

---

## Correzioni obbligatorie

### C1. Sezione 6: il vettore attuale è una direzione di rifiuto, non un test neutro

Lo studio dice che `Qwen3.8-Flash-Next-steering.f32` "non deriva da coppie di prompt" e "serve per testare il
meccanismo, non un comportamento semantico specifico". Il README della stessa cartella dice il contrario:

- `data/experimental-speed-projection/README.md`: "The package's own documentation describes the vector as a
  **refusal-direction projection**: the model declines far fewer requests (it reports 1 of 50 vs 50 of 50 on its
  test set), and removing refusals removes a safety behaviour".
- Stesso README: nell'originale pubblicato i file hanno "refusal" nel nome; agisce sui **layer 4-44**; sposta le
  previsioni anche su testo normale; costa 0,2-0,4 % per token.

**Conseguenza**: tutti i test di steering fatti finora (Fase 7 e Fase 9 di `STEERING_LOG.md`) misurano la rimozione
del rifiuto. Va scritto nella sezione 6 e nel riepilogo della sezione 9.

### C2. Sezioni 2 e 8: Strata non usa il template della chat quando costruisce il vettore

| | ds4 | Strata |
|---|---|---|
| Formattazione del prompt | template della chat del modello (`--profile`, "model shape and chat-template profile", `dir-steering/tools/build_direction.py:123`) + system prompt (`--system`, default "You are a helpful assistant.", riga 133) | testo grezzo: `tok.encode(prompt, parse_special=True)` (`tools/build_steering_vector.py:136`), nessun template né system prompt |
| Posizione letta | ultimo token del prompt formattato (dove l'assistente sta per rispondere) | ultimo token del testo grezzo (`build_steering_vector.py:169-173`) |

La differenza di posizione cambia la direzione catturata probabilmente più della differenza `ffn_out` / residuo della
sezione 8 (**Ipotesi**: da misurare confrontando i due vettori, per esempio con il coseno layer per layer).
Correzione: in `build_steering_vector.py` applicare lo stesso template usato da `tools/steering_ab.py`
(`chat_prompt()`) prima della tokenizzazione, e aggiungere questa riga alla tabella di confronto.

### C3. I test applicano il vettore su layer diversi da quelli scelti dagli autori

- Il `.f32` ha direzioni non nulle sui layer 1-47 (47 su 48, misurato leggendo le norme dal file).
- Gli autori lo usano sui layer 4-44: `--control-vector-layer-range 4 44 --cvec-mode project`
  (`src/program/generate.cpp:544-546`, `docs/DETAILS.md:836`).
- `--dir-steering-file` non imposta un intervallo: senza `--control-vector-layer-range` vale 1 .. L-1
  (`generate.cpp`, `const int first = o.cvec_first <= 0 ? 1 : ...`).

**Conseguenza**: `test_steering_e2e.py`, `test_steering_server.py` e `steering_ab.py` applicano il vettore anche sui
layer 1-3 e 45-47, fuori dalla configurazione validata. Lo studio deve dirlo, e i test dovrebbero passare
`--control-vector-layer-range 4 44` quando vogliono riprodurre la configurazione degli autori.

---

## Imprecisioni da correggere

### I1. Sezione 3: cosa fa la scala

La formula `y = y − scale · dir · dot(dir, y)`:

| scale | Effetto |
|---|---|
| 1 | **azzera** la componente di `y` lungo `dir`: né verso "good" né verso "bad" su quell'asse |
| 2 | **inverte** la componente (riflessione) |
| −1 | **raddoppia** la componente, qualunque sia il suo segno |
| 0 | nessun effetto |

"`scale > 0` sopprime il comportamento good" non è esatto: con scala 1 si toglie l'asse, non si spinge verso "bad".
Va corretta anche la frase "`scale < 0` amplifica la direzione catturata": amplifica la componente che il token ha già.

### I2. Sezione 5: affermazioni senza misura

"Dopo ~20-40 token: over-saturazione stilistica…" e "la direzione è più affidabile nei layer centrali" non hanno né
fonte né misura. Vanno marcate come **Ipotesi**, oppure misurate (vedi M2), oppure citate dal README di ds4 con il
punto preciso.

### I3. Sezione 7: il nome "concept-steering"

L'esempio usa `mlabonne/harmless_alpaca` contro `mlabonne/harmful_behaviors`: la coppia cattura di nuovo una
**direzione di rifiuto**. Va detto, e la cartella di output va chiamata in modo coerente (non `concept-steering`).

---

## Cosa manca

### M1. Scelta dei layer
Descrivere `--control-vector-layer-range A B` e `--cvec-dir single:L` (`generate.cpp:546-550`), e il perché del 4-44
degli autori. Uno sweep sui layer è parte dello studio quanto lo sweep sulla scala.

### M2. Valutazione con numeri
Oggi `steering_ab.py` verifica solo che le uscite siano "diverse". Servono:
- un set di prompt **separato** da quelli usati per costruire il vettore;
- una metrica del comportamento obiettivo (per esempio un tasso, contato da uno script che stampa solo numeri);
- una misura dei danni collaterali su testo normale (perplexity o KL rispetto alla baseline), per scala e per layer.

### M3. Effetti sulle prestazioni di Strata (MoE + MTP)
- Lo steering modifica il residuo, quindi anche **quali esperti vengono scelti**: cambia `decode expert cache hit
  rate` e le letture dal disco (oggi il collo di bottiglia su Gerty: 546 GB letti in 40 minuti, `logs/engine-performance.log`).
- Le bozze MTP non sono steered (sezione 1), quindi con lo steering attivo possono essere accettate meno spesso:
  confrontare `drafts accepted X of Y` con e senza steering.
- Le due righe sono già nel log del motore: basta misurarle a parità di prompt.

### M4. Prompt cache e steering
`docs/DETAILS.md` (Multiple conversations): si riusano solo prefissi "with matching steering mode". Attivare o
spegnere lo steering a metà conversazione costa una rilettura completa (su Gerty: ~8 minuti a 80.000 token,
171 token/s). Va scritto nella sezione sul deployment.

### M5. Protocollo per la conversazione lunga
La sezione 9 la segna come "aperta" ma senza protocollo. Proposta: stessa conversazione reale (~60-80K token),
scala e layer della configurazione degli autori, transient 0/20/100 token; misurare tempo di lettura, token/s,
`drafts accepted`, `hit rate`, eventuali `Xid` e la stabilità della risposta.

### M6. Sezione "Non verificato"
Aggiungere alla fine dello studio l'elenco di cosa non è stato misurato (oggi almeno: C2, I2, M2, M3, M5).
