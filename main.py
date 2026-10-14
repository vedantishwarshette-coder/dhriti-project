from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import time

from quality.scorer import evaluate_response
from watchdog.drift import DriftWatchdog
from healing.pipeline import SelfHealingPipeline

app = FastAPI(
    title="DHRITI AI Platform Gateway",
    description="Full MLOps Platform with Cache, Quality Scorer, Drift Watchdog, and Self-Healing",
    version="1.0.0"
)

# Initialize all core components
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

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "DHRITI Full MLOps Pipeline Active"}

@app.post("/v1/generate", response_model=InferenceResponse)
def generate_text(request: InferenceRequest):
    start_time = time.perf_counter()
    
    clean_prompt = request.prompt.strip().lower()
    if not clean_prompt:
        raise HTTPException(status_code=400, detail="Prompt empty nahi ho sakta!")
    
    # 1. Cache Check
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
    
    # 2. Generate response & evaluate quality
    generated_answer = f"[{request.prompt} ka AI response - generated live!]"
    score = evaluate_response(request.prompt, generated_answer)
    
    # 3. Watchdog records score
    watchdog.add_score(score)
    is_drift = watchdog.check_drift()
    
    healing_triggered = False
    # 4. If drift detected, trigger self-healing!
    if is_drift:
        healing_triggered = healing_pipeline.trigger_fine_tuning(request.model_name)
    
    # Save to cache
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