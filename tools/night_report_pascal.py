"""Summarize saved numerical evidence without reading prompt or answer text."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from collections import Counter

from tools.test_upstream_pascal import replace_arg
from tools.night_pascal import matrix


def summarize(root):
    summary = json.loads((root / "summary.json").read_text())
    cases = list(summary["cases"])
    recorded = {case["name"] for case in cases}
    for case in matrix():
        if case["name"] in recorded:
            continue
        supervisor_path = root / (case["name"] + ".supervisor.json")
        if supervisor_path.exists():
            supervisor = json.loads(supervisor_path.read_text())
            status = supervisor.get("status")
            cases.append({**case, "status": "INCOMPLETE_EVIDENCE",
                          "error": f"No completed summary record; saved supervisor status: {status}. "
                                   "Execution interrupted or not finalized; not validated"})
        else:
            cases.append({**case, "status": "NOT_RUN",
                          "error": "No completed supervisor record; not validated"})
    rows = []
    for case in cases:
        path = root / (case["name"] + ".json")
        data = json.loads(path.read_text()) if path.exists() else {}
        supervisor_path = root / (case["name"] + ".supervisor.json")
        supervisor = json.loads(supervisor_path.read_text()) if supervisor_path.exists() else {}
        requests = data.get("requests", [])
        row = {
            "name": case["name"], "status": case["status"], "stage": case.get("stage"),
            "requests": len(requests), "requested": data.get("requested_requests"),
            "prompt_max": max((r["prompt_tokens"] for r in requests), default=0),
            "attempted_prompt": data.get("current_request", {}).get("prompt_tokens"),
            "known_correct": sum(r.get("known_answer_ok", False) for r in requests),
            "cancelled": sum(r.get("cancelled", False) for r in requests),
            "startup_s": data.get("startup_s"),
            "error": data.get("error") or case.get("error") or supervisor.get("error"),
            "fresh_parity": [r["fresh_parity"] for r in requests if "fresh_parity" in r],
            "two_pass_requests": sum(r.get("engine_passes", 1) >= 2 for r in requests),
            "http_stream_gate": data.get("http_stream_gate", False),
            "http_completion_tokens": data.get("http_completion_tokens"),
            "performance_only": data.get("performance_only", False),
            "engine_info": data.get("engine_info", {}),
        }
        measured = [r for r in requests[1:] if not r["cancelled"]]
        elapsed = [r["elapsed_s"] for r in measured]
        decode = [r["metrics"]["generated"] * 1000 / r["metrics"]["decode_ms"] for r in measured
                  if r["metrics"].get("decode_ms", 0) > 0]
        row["median_elapsed_s"] = statistics.median(elapsed) if elapsed else None
        row["median_decode_tok_s"] = statistics.median(decode) if decode else None
        row["rss_growth_kib"] = data.get("post_warmup_rss_growth_kib")
        if case["status"] == "PASS" and (data.get("status") != "OK" or len(requests) != data.get("requested_requests")):
            row["status"] = "INCOMPLETE_EVIDENCE"
        rows.append(row)
    return rows


def recommend(root, rows):
    by_name = {row["name"]: row for row in rows}
    def passed(name):
        return by_name.get(name, {}).get("status") == "PASS"
    def wrapped(name):
        return passed(name) and by_name[name].get("two_pass_requests", 0) > 0
    def http_passed(name):
        return (passed(name) and by_name[name].get("http_stream_gate", False) and
                (by_name[name].get("http_completion_tokens") or 0) > 0)
    def parity_passed(name):
        return passed(name) and by_name[name].get("fresh_parity") == [True]
    def soak_passed(name):
        row = by_name.get(name, {})
        return (passed(name) and row.get("requests") == 100 and row.get("cancelled", 0) > 0 and
                row.get("known_correct", 0) + row.get("cancelled", 0) == 100 and
                row.get("rss_growth_kib") is not None and row["rss_growth_kib"] <= 512 * 1024)
    common_gate = passed("reasoning-arithmetic") and http_passed("http-thinking")
    normal_ok = (common_gate and passed("candidate-controlled") and passed("candidate-explicit-off") and
                 all(passed(f"context-{length}") for length in (4096, 16384, 24576, 30000)) and
                 passed("thinking-long") and soak_passed("soak100") and
                 wrapped("thinking-forced-wrap") and parity_passed("conversation-reuse"))
    reverse_ok = (common_gate and passed("reverse-order") and soak_passed("reverse-soak100") and
                  passed("reverse-context30k") and
                  wrapped("reverse-thinking") and parity_passed("reverse-conversation"))
    serial_ok = (passed("serial") and passed("reasoning-arithmetic") and http_passed("serial-thinking") and
                 wrapped("serial-thinking") and passed("serial-context30k") and
                 parity_passed("serial-conversation") and soak_passed("serial-soak100"))
    order_samples = {}
    for arm in ("normal", "reverse"):
        samples = [row.get("median_decode_tok_s") for row in rows
                   if row["name"].startswith("order-performance-") and row["name"].endswith(arm)
                   and row["status"] == "PASS" and row.get("median_decode_tok_s") is not None]
        order_samples[arm] = statistics.median(samples) if len(samples) >= 3 else None
    selected = None
    if normal_ok:
        selected = "normal"
    if reverse_ok and (selected is None or
                       (order_samples["reverse"] is not None and order_samples["normal"] is not None
                        and order_samples["reverse"] > order_samples["normal"])):
        selected = "reverse"
    if selected is None and serial_ok:
        selected = "serial"
    result = {"qualified_order": selected, "order_decode_medians": order_samples,
              "normal_gate": normal_ok, "reverse_gate": reverse_ok, "serial_gate": serial_ok}
    if selected:
        config = json.loads((root / "input.config.json").read_text())
        config["args"] = list(config["args"])
        for flag in ("--expert-profile-save", "--expert-profile-save-every"):
            while flag in config["args"]:
                index = config["args"].index(flag)
                del config["args"][index:index + 2]
        config["gpu"] = [0, 1] if selected == "reverse" else [1, 0]
        config["gpu_order"] = config["vram_reserve"] = "as_given"
        config["env"].update(STRATA_DF_BRANCH="0", STRATA_MTP_CATCHUP_ALL="0", STRATA_PIN_GUARD="2")
        config["reasoning_budget_tokens"] = 1024
        config["reasoning_min_answer_tokens"] = 2048
        replace_arg(config["args"], "--pipeline-windows", "0" if selected == "serial" else "2")
        replace_arg(config["args"], "--max-context", "32768")
        replace_arg(config["args"], "--layer-split", "22")
        replace_arg(config["args"], "--resident-budget-gib", "16")
        replace_arg(config["args"], "--vram-reserve-mib", "1800" if selected == "reverse" else "512")
        replace_arg(config["args"], "--vram-reserve-later-mib", "512" if selected == "reverse" else "1800")
        config.pop("log", None)
        path = root / "recommended-32k.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        result["config"] = str(path)
        result["qualification_limit"] = "Synthetic known-answer and soak gates only; not a general correctness guarantee"
    pairs = []
    for pair in range(5):
        arms = [by_name.get(f"performance-{pair}-{arm}", {}) for arm in ("stable", "candidate")]
        if all(row.get("status") == "PASS" and (row.get("median_decode_tok_s") or 0) > 0 for row in arms):
            pairs.append((arms[1]["median_decode_tok_s"] / arms[0]["median_decode_tok_s"] - 1) * 100)
    result["matched_speed_pairs"] = len(pairs)
    result["matched_speed_change_percent"] = statistics.median(pairs) if len(pairs) == 5 else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(args.output_dir)
    recommendation = recommend(args.output_dir, rows)
    baseline = json.loads(args.baseline.read_text())
    protected = []
    for name, expected in baseline["files"].items():
        with Path(name).open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        protected.append({"path": name, "unchanged": digest == expected})
    result = {"cases": rows, "recommendation": recommendation, "protected": protected,
              "generated_unix": time.time(), "started_unix": baseline["started_unix"],
              "deadline_unix": baseline["deadline_unix"],
              "status_counts": dict(Counter(row["status"] for row in rows)),
              "all_protected_unchanged": all(item["unchanged"] for item in protected)}
    validation_path = args.output_dir / "final-validation.json"
    if validation_path.exists():
        result["final_validation"] = json.loads(validation_path.read_text())
    reboot_path = args.output_dir / "reboot-diagnosis.json"
    if reboot_path.exists():
        result["reboot_diagnosis"] = json.loads(reboot_path.read_text())
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    completed_prompt_max = max((row["prompt_max"] for row in rows if row["status"] == "PASS"), default=0)
    interrupted = [row["name"] for row in rows if row["status"] == "INCOMPLETE_EVIDENCE"]
    lines = ["# Strata Pascal: rapporto dei test notturni", "",
             "Build candidata: backport selettivo upstream, non merge completo.", "",
             "## Esito", "",
             f"Conteggi: {result['status_counts']}.",
             "Nessuna configurazione e' qualificata se mancano soak, conversazione e parity.",
             f"Casi con evidenza incompleta: {interrupted}. Non sono PASS; "
             "l'interruzione non e' attribuita al deadline senza prova.",
             f"Il prompt massimo nei casi PASS e' {completed_prompt_max} token. "
             "La tabella distingue casi completati, interrotti e mai avviati.",
             "",
             "| Caso | Stato | Richieste | Prompt massimo reale | Risposte corrette | Cancellazioni |",
             "|---|---|---:|---:|---:|---:|"]
    lines += [f"| {r['name']} | {r['status']} | {r['requests']} | {r['prompt_max']} | "
              f"{r['known_correct']} | {r['cancelled']} |" for r in rows]
    lines += ["", "## Raccomandazione", "", f"Configurazione qualificata: {recommendation['qualified_order']}.",
              "Se il valore e' None, nessuna configurazione ha superato tutti i gate: "
              "non viene proposta come affidabile ne' installata.",
              "Budget iniziale da provare: reasoning 1024, riserva risposta 2048, "
              "max_tokens almeno 3328; con contesti quasi pieni la richiesta viene rifiutata "
              "invece di esaurire lo spazio durante il thinking. Nessuna garanzia per risposte arbitrarie.",
              "Dati numerici, velocita', configurazioni effettive e errori sono in report.json "
              "e nei file per caso.",
              "stable-controlled usava i graph branches predefiniti del vecchio engine: "
              "non e' un confronto equo delle prestazioni.",
              "adaptive-off: prima esecuzione fallita per un controllo errato del runner sul "
              "salvataggio del profilo, non per la risposta del modello. adaptive-off-checked "
              "usa il controllo corretto.",
              "adaptive-blocking iniziale: interrotto dal timeout del monitor GPU; "
              "cleanup eseguito sui soli processi del test. Non e' un PASS.",
              "I budget 1024/2048/4096 possono non venire raggiunti dai quesiti semplici. "
              "Il wrap forzato verifica il secondo passaggio reale dell'engine.",
              "I contesti oltre 32K usano KV streaming, non KV interamente residente.",
              "I gate sintetici non garantiscono correttezza per conversazioni arbitrarie.",
              "La modalita' seriale con GPU fisiche [1,0] e pipeline 0 e' solo un candidato "
              "per ulteriori prove se non supera tutti i gate: non un fallback gia' certificato. "
              "Non vietare configurazioni sulla sola base degli errori del monitor/runner: "
              "valutare separatamente le ripetizioni con nome distinto.",
              "Non ridurre max_tokens sotto il minimo necessario: il server rifiuta "
              "esplicitamente lo spazio insufficiente. La riserva non garantisce il "
              "completamento di ogni risposta e non corregge gli stalli CUDA.",
              "", "## Protezioni", "",
              f"Tutti i file protetti invariati: {result['all_protected_unchanged']}."]
    lines += ["", "## Errori e prove incomplete", ""]
    lines += [f"- {r['name']}: {r['status']}; {r.get('error') or 'vedere il log supervisor/engine'}"
              for r in rows if r["status"] != "PASS"]
    if "final_validation" in result:
        lines += ["", "## Verifiche finali e risorse", "",
                  "Metadati delle verifiche effettivamente eseguite; nessun test mancante "
                  "viene conteggiato come superato.", "",
                  "```json", json.dumps(result["final_validation"], indent=2), "```"]
    if "reboot_diagnosis" in result:
        diagnosis = result["reboot_diagnosis"]
        lines += ["", "## Riavvio della macchina", "",
                  f"Nuovo boot: {diagnosis['current_boot_started_local']}. "
                  f"Ultimo campione salvato: {diagnosis['last_saved_sample_local']}.",
                  "Riavvio non ordinato confermato: ha interrotto il test, non il limite "
                  "delle sette ore. La causa del reset non e' provata. Nessun errore "
                  "MCE/EDAC, OOM, panic o NVIDIA Xid trovato nel journal disponibile.",
                  "Temperature GPU registrate al massimo 70/59 C; temperatura CPU "
                  "non registrata. Overclock CPU/RAM, alimentazione e un crash "
                  "kernel/driver non registrato restano ipotesi.",
                  "Dettagli e limiti dell'evidenza in reboot-diagnosis.json."]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"cases": len(rows), "protected_unchanged": result["all_protected_unchanged"],
                      "qualified_order": recommendation["qualified_order"]}))
    if not result["all_protected_unchanged"]:
        raise SystemExit("Protected artifacts changed")


if __name__ == "__main__":
    main()
