from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import uuid

# Fix TensorFlow/PyTorch issues
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.generation import AudioGenerator
from src.evaluation import AudioEvaluator
from src.rag_enhancer import PromptEnhancer

app = FastAPI(title="Text-to-Audio GenAI Dashboard")

generator = AudioGenerator()
evaluator = AudioEvaluator()
rag = PromptEnhancer()

class AudioRequest(BaseModel):
    prompt: str
    audio_length: float = 3.0

class AudioResponse(BaseModel):
    audio_id: str
    original_prompt: str
    enhanced_prompt: str
    audio_path: str
    poas_score: float

@app.post("/generate", response_model=AudioResponse)
async def generate_audio(request: AudioRequest):
    try:
        audio_id = str(uuid.uuid4())[:8]
        
        enhanced_prompt = rag.enhance(request.prompt)
        
        audio_filename = f"outputs/audio/{audio_id}.wav"
        generator.generate(
            prompt=enhanced_prompt,
            output_path=audio_filename,
            num_inference_steps=20,
            audio_length_in_s=request.audio_length
        )
        
        poas_score = evaluator.evaluate_clap(enhanced_prompt, audio_filename)
        
        return AudioResponse(
            audio_id=audio_id,
            original_prompt=request.prompt,
            enhanced_prompt=enhanced_prompt,
            audio_path=audio_filename,
            poas_score=poas_score
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    import pandas as pd
    
    results_csv = "outputs/results_table.csv"
    if not os.path.exists(results_csv):
        return {"message": "No results yet. Generate audio first."}
    
    df = pd.read_csv(results_csv)
    
    return {
        "total_samples": len(df),
        "mean_poas": float(df['poas_score'].mean()) if len(df) > 0 else 0,
        "domain_breakdown": df.groupby('domain')['poas_score'].mean().to_dict(),
        "latest_results": df.tail(10).to_dict(orient="records")
    }

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GenAI Text-to-Audio Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
            h1 { color: #2c3e50; text-align: center; }
            .card { background: white; padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            input, select { padding: 10px; width: 70%; border: 1px solid #ddd; border-radius: 5px; }
            button { padding: 10px 20px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer; margin-left: 10px; }
            button:hover { background: #2980b9; }
            .result { background: #ecf0f1; padding: 10px; margin: 10px 0; border-radius: 5px; }
            audio { width: 100%; margin-top: 10px; }
            .metric { display: inline-block; margin: 10px 20px; }
            .metric-value { font-size: 24px; font-weight: bold; color: #3498db; }
            #loading { display: none; text-align: center; padding: 20px; }
        </style>
    </head>
    <body>
        <h1>🎵 GenAI Text-to-Audio Dashboard</h1>
        
        <div class="card">
            <h2>Generate Audio</h2>
            <input type="text" id="prompt" placeholder="Enter prompt (e.g., 'rain sound', 'piano melody')" value="rain sound">
            <input type="number" id="length" placeholder="Duration (seconds)" value="3" style="width: 15%;">
            <button onclick="generateAudio()">Generate</button>
            <div id="loading">Generating audio... Please wait...</div>
        </div>
        
        <div class="card" id="result-card" style="display:none;">
            <h2>Generated Audio</h2>
            <div class="result">
                <p><strong>Original Prompt:</strong> <span id="orig-prompt"></span></p>
                <p><strong>Enhanced Prompt:</strong> <span id="enh-prompt"></span></p>
                <p><strong>POAS Score:</strong> <span id="poas"></span></p>
                <audio id="audio-player" controls></audio>
            </div>
        </div>
        
        <div class="card">
            <h2>Metrics Dashboard</h2>
            <button onclick="loadMetrics()">Refresh Metrics</button>
            <div id="metrics-display">
                <div class="metric">Total Samples: <div class="metric-value" id="total-samples">-</div></div>
                <div class="metric">Mean POAS: <div class="metric-value" id="mean-poas">-</div></div>
            </div>
            <h3>Domain Breakdown</h3>
            <div id="domain-breakdown"></div>
            <h3>Recent Results</h3>
            <div id="recent-results"></div>
        </div>
        
        <script>
            async function generateAudio() {
                const prompt = document.getElementById('prompt').value;
                const length = parseFloat(document.getElementById('length').value);
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result-card').style.display = 'none';
                
                try {
                    const response = await fetch('/generate', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({prompt: prompt, audio_length: length})
                    });
                    const data = await response.json();
                    
                    document.getElementById('orig-prompt').textContent = data.original_prompt;
                    document.getElementById('enh-prompt').textContent = data.enhanced_prompt;
                    document.getElementById('poas').textContent = data.poas_score.toFixed(4);
                    document.getElementById('audio-player').src = data.audio_path;
                    document.getElementById('result-card').style.display = 'block';
                    
                    loadMetrics();
                } catch (e) {
                    alert('Error: ' + e.detail);
                }
                
                document.getElementById('loading').style.display = 'none';
            }
            
            async function loadMetrics() {
                try {
                    const response = await fetch('/metrics');
                    const data = await response.json();
                    
                    document.getElementById('total-samples').textContent = data.total_samples || 0;
                    document.getElementById('mean-poas').textContent = (data.mean_poas || 0).toFixed(4);
                    
                    const breakdown = data.domain_breakdown || {};
                    document.getElementById('domain-breakdown').innerHTML = 
                        Object.entries(breakdown).map(([d, v]) => `<span style="margin-right:15px;">${d}: <b>${v.toFixed(4)}</b></span>`).join('');
                    
                    const results = data.latest_results || [];
                    document.getElementById('recent-results').innerHTML = results.map(r => 
                        `<div class="result"><b>${r.domain}</b>: ${r.original_prompt} → POAS: ${r.poas_score.toFixed(4)}</div>`
                    ).join('');
                } catch (e) {
                    console.log('No metrics yet');
                }
            }
            
            loadMetrics();
        </script>
    </body>
    </html>
    """
