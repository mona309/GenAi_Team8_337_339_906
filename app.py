from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os, uuid
from dotenv import load_dotenv

load_dotenv()

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.generation import AudioGenerator
from src.evaluation import AudioEvaluator
from src.rag_enhancer import PromptEnhancer

app = Flask(__name__)

generator = AudioGenerator()
evaluator = AudioEvaluator()
rag       = PromptEnhancer()

results_store = []

CSV_DIR             = "outputs"
RESULTS_CSV         = os.path.join(CSV_DIR, "results_table.csv")
DOMAIN_TRANSFER_CSV = os.path.join(CSV_DIR, "domain_transfer.csv")


# ── CSV helpers ────────────────────────────────────────────────────────────────

def _flush_csvs():
    """
    Write results_table.csv and domain_transfer.csv from results_store.
    Called after every /generate and after /results DELETE.

    results_table.csv:    id, domain, original_prompt, enhanced_prompt,
                          poas_score, clap_score, semantic_score, audio_file
    domain_transfer.csv:  domain, samples, mean_poas, std_poas,
                          cri_local, cdts_local
    """
    os.makedirs(CSV_DIR, exist_ok=True)

    # ── results_table.csv ─────────────────────────────────────────────────
    results_df = pd.DataFrame([
        {
            "id":               r["id"],
            "domain":           r["domain"],
            "original_prompt":  r["original_prompt"],
            "enhanced_prompt":  r["enhanced_prompt"],
            "poas_score":       round(r["poas_score"],     6),
            "clap_score":       round(r.get("clap_score",     0.0), 6),
            "semantic_score":   round(r.get("semantic_score", 0.0), 6),
            "audio_file":       r["audio_file"],
        }
        for r in results_store
    ])
    results_df.to_csv(RESULTS_CSV, index=False)

    # ── domain_transfer.csv ───────────────────────────────────────────────
    if not results_store:
        pd.DataFrame(columns=[
            "domain", "samples", "mean_poas", "std_poas",
            "cri_local", "cdts_local",
        ]).to_csv(DOMAIN_TRANSFER_CSV, index=False)
        return

    all_scores  = [r["poas_score"] for r in results_store]
    global_mean = float(np.mean(all_scores))
    global_std  = float(np.std(all_scores)) or 1.0

    buckets: dict = {}
    for r in results_store:
        buckets.setdefault(r["domain"], []).append(r["poas_score"])

    rows = []
    for domain, scores in buckets.items():
        arr    = np.array(scores)
        mean_p = float(np.mean(arr))
        std_p  = float(np.std(arr))
        rows.append({
            "domain":     domain,
            "samples":    len(scores),
            "mean_poas":  round(mean_p, 6),
            "std_poas":   round(std_p,  6),
            "cri_local":  round(std_p,  6),
            "cdts_local": round((mean_p - global_mean) / global_std, 6),
        })

    pd.DataFrame(rows).to_csv(DOMAIN_TRANSFER_CSV, index=False)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/generate", methods=["POST"])
def generate():
    data      = request.json
    prompt    = data["prompt"]
    domain    = data["domain"]
    use_rag   = data.get("use_rag", True)
    steps     = data.get("num_inference_steps", 120)
    audio_len = data.get("audio_length_in_s", 8.0)
    gender    = data.get("gender", "female")
    emotion   = data.get("emotion", "neutral")   # new: passed from dashboard

    # ── RAG enhancement ────────────────────────────────────────────────────
    # rag.enhance() now calls fetch_retrieved_context() internally, so
    # the KB keyword retrieval is actually used.  Speech passes through
    # unchanged (SpeechT5 is sensitive to added descriptors).
    enhanced = rag.enhance(prompt, domain) if use_rag else prompt

    run_id     = str(uuid.uuid4())[:8]
    audio_file = f"outputs/audio/{domain}_{run_id}.wav"

    # ── Generation ─────────────────────────────────────────────────────────
    generator.generate(
        prompt=enhanced,
        output_path=audio_file,
        domain=domain,
        gender=gender,
        emotion=emotion,          # passed through to SpeechT5
        num_inference_steps=steps,
        audio_length_in_s=audio_len,
    )

    # ── Evaluation ─────────────────────────────────────────────────────────
    # Previously app.py called evaluator.evaluate_clap() directly.
    # That bypassed the 0.90-weight ASR term for speech domains, making
    # speech POAS scores meaningless (pure CLAP on TTS audio).
    # Now we call evaluate_poas() which handles domain routing correctly.
    scores = evaluator.evaluate_poas(
        text_prompt=enhanced,
        audio_path=audio_file,
        domain=domain,
    )

    record = dict(
        id=run_id,
        domain=domain,
        original_prompt=prompt,
        enhanced_prompt=enhanced,
        poas_score=scores["poas_score"],
        clap_score=scores["clap_score"],
        semantic_score=scores["semantic_score"],
        audio_file=audio_file,
        audio_url=f"/audio/{domain}_{run_id}.wav",
        rag_used=use_rag,
        emotion=emotion,
    )
    results_store.append(record)
    _flush_csvs()
    return jsonify(record)


@app.route("/metrics")
def metrics():
    if not results_store:
        return jsonify(avg_poas=None, cri_variance=None, run_count=0, by_domain={})

    scores  = [r["poas_score"] for r in results_store]
    avg     = sum(scores) / len(scores)
    buckets: dict = {}
    for r in results_store:
        buckets.setdefault(r["domain"], []).append(r["poas_score"])

    by_domain = {d: sum(v) / len(v) for d, v in buckets.items()}
    means     = list(by_domain.values())
    gm        = sum(means) / len(means)
    cri       = sum((m - gm) ** 2 for m in means) / len(means) if len(means) >= 2 else None

    return jsonify(
        avg_poas=round(avg, 6),
        cri_variance=round(cri, 6) if cri else None,
        run_count=len(results_store),
        by_domain=by_domain,
    )


@app.route("/csv-data")
def csv_data():
    """Serve both CSVs as JSON for dashboard chart rendering."""
    results_rows = []
    domain_rows  = []

    if os.path.exists(RESULTS_CSV):
        try:
            results_rows = pd.read_csv(RESULTS_CSV).to_dict(orient="records")
        except Exception:
            pass

    if os.path.exists(DOMAIN_TRANSFER_CSV):
        try:
            domain_rows = pd.read_csv(DOMAIN_TRANSFER_CSV).to_dict(orient="records")
        except Exception:
            pass

    return jsonify(results=results_rows, domain_transfer=domain_rows)


@app.route("/results", methods=["DELETE"])
def clear():
    results_store.clear()
    _flush_csvs()
    return "", 204


@app.route("/audio/<path:filename>")
def serve_audio(filename):
    return send_file(os.path.join("outputs/audio", filename), mimetype="audio/wav")


if __name__ == "__main__":
    app.run(debug=True, port=5000)