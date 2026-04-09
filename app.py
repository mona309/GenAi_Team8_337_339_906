from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os, uuid
from dotenv import load_dotenv

load_dotenv()

#added for prompt input
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.generation import AudioGenerator
from src.evaluation import AudioEvaluator
from src.rag_enhancer import PromptEnhancer

app = Flask(__name__)

generator = AudioGenerator()
evaluator = AudioEvaluator()
rag = PromptEnhancer()

results_store = []

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data["prompt"]
    domain = data["domain"]
    use_rag = data.get("use_rag", True)
    steps = data.get("num_inference_steps", 120)
    audio_len = data.get("audio_length_in_s", 8.0)
    gender = data.get("gender", "female")

    enhanced = rag.enhance(prompt, domain) if use_rag else prompt

    run_id = str(uuid.uuid4())[:8]
    audio_file = f"outputs/audio/{domain}_{run_id}.wav"
    generator.generate(
        prompt=enhanced,
        output_path=audio_file,
        domain=domain,
        gender=gender,
        num_inference_steps=steps,
        audio_length_in_s=audio_len
    )
    poas = float(evaluator.evaluate_clap(enhanced, audio_file))

    record = dict(id=run_id, domain=domain, original_prompt=prompt, enhanced_prompt=enhanced, poas_score=poas, audio_file=audio_file, audio_url=f"/audio/{domain}_{run_id}.wav", rag_used=use_rag)
    
    results_store.append(record)
    return jsonify(record)

@app.route("/metrics")
def metrics():
    if not results_store:
        return jsonify(avg_poas=None, cri_variance=None, run_count=0, by_domain={})
    scores = [r["poas_score"] for r in results_store]
    avg = sum(scores) / len(scores)
    buckets = {}
    for r in results_store:
        buckets.setdefault(r["domain"], []).append(r["poas_score"])
    by_domain = {d: sum(v)/len(v) for d, v in buckets.items()}
    means = list(by_domain.values())
    gm = sum(means) / len(means)
    cri = sum((m-gm)**2 for m in means)/len(means) if len(means) >= 2 else None
    return jsonify(avg_poas=round(avg,6), cri_variance=round(cri,6) if cri else None,
                run_count=len(results_store), by_domain=by_domain)

@app.route("/results", methods=["DELETE"])
def clear():
    results_store.clear()
    return "", 204

@app.route("/audio/<path:filename>")
def serve_audio(filename):
    return send_file(os.path.join("outputs/audio", filename), mimetype="audio/wav")

if __name__ == "__main__":
    app.run(debug=True, port=5000)