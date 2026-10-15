from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import time

from quality.scorer import evaluate_response
from watchdog.drift import DriftWatchdog
from healing.pipeline import SelfHealingPipeline

app = FastAPI(
    title="DHRITI AI Platform Gateway",
    description="Full MLOps Platform with Built-in Dashboard",
    version="1.0.0"
)

# Initialize core components
CACHE_STORE = {}
watchdog = DriftWatchdog(threshold=0.6)
healing_pipeline = SelfHealingPipeline()

class InferenceRequest(BaseModel):
    prompt: str
    model_name: Optional[str] = "dhriti-model-v1"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 100

class InferenceResponse(BaseModel):
    request_id: str
    model_used: str
    response_text: str
    quality_score: float
    cache_hit: bool
    drift_detected: bool
    healing_triggered: bool
    latency_ms: float

# Built-in Professional Dashboard HTML
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DHRITI MLOps Control Center</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        body { background-color: #0f172a; color: #f8fafc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .navbar { background-color: #1e293b; border-bottom: 1px solid #334155; }
        .card { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; color: #f8fafc; }
        .metric-title { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; }
        .metric-value { font-size: 1.8rem; font-weight: 700; }
        .badge-success { background-color: #059669; }
        .badge-warning { background-color: #d97706; }
        .badge-danger { background-color: #dc2626; }
        .form-control, .form-select { background-color: #0f172a; border: 1px solid #334155; color: #f8fafc; }
        .form-control:focus, .form-select:focus { background-color: #0f172a; color: #f8fafc; border-color: #38bdf8; box-shadow: 0 0 0 0.25rem rgba(56, 189, 248, 0.25); }
        .btn-custom { background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%); color: white; font-weight: 600; border: none; }
        .btn-custom:hover { background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 100%); color: white; }
        .response-box { background-color: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 15px; font-family: monospace; white-space: pre-wrap; }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark px-4 py-3">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1 fs-4 fw-bold">
                <i class="fa-solid fa-shield-halved text-info me-2"></i> DHRITI <span class="badge bg-primary fs-6 ms-2">v1.0 Platform</span>
            </span>
            <div class="d-flex align-items-center">
                <span class="badge bg-success me-3"><i class="fa-solid fa-circle-check me-1"></i> System Online</span>
                <small class="text-secondary">CSE Final Year Project</small>
            </div>
        </div>
    </nav>

    <div class="container-fluid px-4 py-4">
        <div class="row g-3 mb-4">
            <div class="col-md-3">
                <div class="card p-3 shadow-sm">
                    <div class="metric-title"><i class="fa-solid fa-bolt text-warning me-2"></i> Latency Speed</div>
                    <div class="metric-value text-info mt-2" id="metric-latency">0.00 ms</div>
                    <small class="text-secondary">Response processing time</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3 shadow-sm">
                    <div class="metric-title"><i class="fa-solid fa-star text-warning me-2"></i> Quality Score</div>
                    <div class="metric-value text-success mt-2" id="metric-quality">1.00</div>
                    <small class="text-secondary">Evaluation Scorer metric</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3 shadow-sm">
                    <div class="metric-title"><i class="fa-solid fa-database text-primary me-2"></i> Semantic Cache</div>
                    <div class="metric-value mt-2" id="metric-cache"><span class="badge badge-warning">Awaiting Request</span></div>
                    <small class="text-secondary">Cache Hit / Miss Status</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3 shadow-sm">
                    <div class="metric-title"><i class="fa-solid fa-heart-pulse text-danger me-2"></i> Drift & Self-Healing</div>
                    <div class="metric-value mt-2" id="metric-drift"><span class="badge badge-success">Healthy</span></div>
                    <small class="text-secondary">Watchdog Monitoring</small>
                </div>
            </div>
        </div>

        <div class="row g-4">
            <div class="col-md-6">
                <div class="card p-4 shadow-sm h-100">
                    <h5 class="fw-bold mb-3"><i class="fa-solid fa-terminal text-info me-2"></i> Inference Request Terminal</h5>
                    <div class="mb-3">
                        <label class="form-label fw-semibold">Target Model</label>
                        <select class="form-select" id="modelSelect">
                            <option value="dhriti-model-v1">dhriti-model-v1 (Production Champion)</option>
                            <option value="llama-3-8b-instruct">Llama-3-8B-Instruct (Challenger)</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label class="form-label fw-semibold">User Prompt / Query</label>
                        <textarea class="form-control" id="promptInput" rows="5" placeholder="Type your query here to test the DHRITI Gateway pipeline..."></textarea>
                    </div>
                    <button class="btn btn-custom w-100 py-2 fs-6" id="submitBtn" onclick="sendInference()">
                        <i class="fa-solid fa-paper-plane me-2"></i> Send Request through DHRITI Gateway
                    </button>
                </div>
            </div>

            <div class="col-md-6">
                <div class="card p-4 shadow-sm h-100">
                    <h5 class="fw-bold mb-3"><i class="fa-solid fa-chart-line text-info me-2"></i> Gateway Telemetry Output</h5>
                    <label class="form-label fw-semibold text-secondary">Model Response</label>
                    <div class="response-box text-success mb-3" id="responseText" style="min-height: 120px;">[Waiting for request...]</div>
                    <label class="form-label fw-semibold text-secondary">Full Telemetry JSON Payload</label>
                    <div class="response-box text-info" id="rawJson" style="min-height: 150px;">{}</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function sendInference() {
            const prompt = document.getElementById('promptInput').value.trim();
            const model = document.getElementById('modelSelect').value;
            const btn = document.getElementById('submitBtn');
            if (!prompt) { alert("Please enter a prompt!"); return; }
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span> Processing...';
            try {
                const response = await fetch('/v1/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt, model_name: model })
                });
                const data = await response.json();
                document.getElementById('metric-latency').innerText = data.latency_ms + " ms";
                document.getElementById('metric-quality').innerText = data.quality_score;
                const cacheEl = document.getElementById('metric-cache');
                if (data.cache_hit) {
                    cacheEl.innerHTML = '<span class="badge badge-success">CACHE HIT</span>';
                } else {
                    cacheEl.innerHTML = '<span class="badge badge-warning">CACHE MISS</span>';
                }
                const driftEl = document.getElementById('metric-drift');
                if (data.drift_detected) {
                    driftEl.innerHTML = '<span class="badge badge-danger">DRIFT ALERT!</span>';
                } else {
                    driftEl.innerHTML = '<span class="badge badge-success">Healthy</span>';
                }
                document.getElementById('responseText').innerText = data.response_text;
                document.getElementById('rawJson').innerText = JSON.stringify(data, null, 2);
            } catch (err) {
                alert("Error connecting to DHRITI Gateway!");
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i class="fa-solid fa-paper-plane me-2"></i> Send Request through DHRITI Gateway';
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return DASHBOARD_HTML

@app.post("/v1/generate", response_model=InferenceResponse)
def generate_text(request: InferenceRequest):
    start_time = time.perf_counter()
    clean_prompt = request.prompt.strip().lower()
    if not clean_prompt:
        raise HTTPException(status_code=400, detail="Prompt empty nahi ho sakta!")
    
    if clean_prompt in CACHE_STORE:
        cached_data = CACHE_STORE[clean_prompt]
        latency = (time.perf_counter() - start_time) * 1000
        return InferenceResponse(
            request_id="req_cache_hit",
            model_used=request.model_name,
            response_text=cached_data["response"],
            quality_score=cached_data["score"],
            cache_hit=True,
            drift_detected=watchdog.check_drift(),
            healing_triggered=False,
            latency_ms=round(latency, 2)
        )
    
    generated_answer = f"[{request.prompt} ka AI response - generated live!]"
    score = evaluate_response(request.prompt, generated_answer)
    
    watchdog.add_score(score)
    is_drift = watchdog.check_drift()
    
    healing_triggered = False
    if is_drift:
        healing_triggered = healing_pipeline.trigger_fine_tuning(request.model_name)
    
    CACHE_STORE[clean_prompt] = {
        "response": generated_answer,
        "score": score
    }
    
    latency = (time.perf_counter() - start_time) * 1000
    return InferenceResponse(
        request_id="req_new_gen",
        model_used=request.model_name,
        response_text=generated_answer,
        quality_score=score,
        cache_hit=False,
        drift_detected=is_drift,
        healing_triggered=healing_triggered,
        latency_ms=round(latency, 2)
    )   