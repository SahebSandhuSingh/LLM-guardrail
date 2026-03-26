import sys
import os

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from models.schemas import AnalyzeRequest
from pipeline import (
    Layer0Normalizer,
    Layer1Classifiers,
    PGMLayer,
    SessionRiskLayer,
    ExplainabilityLayer,
    ResponseEngine,
)

app = FastAPI(
    title="SafeLayer",
    description="Multi-layer AI safety analysis pipeline",
    version="0.1.0",
)

# Instantiate pipeline layers once at startup
layer0 = Layer0Normalizer()
layer1 = Layer1Classifiers()
layer2 = PGMLayer()
layer3 = SessionRiskLayer()
layer4 = ExplainabilityLayer()
response_engine = ResponseEngine()

PIPELINE = [layer0, layer1, layer2, layer3, layer4, response_engine]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    pipeline_state: dict = {
        "session_id": request.session_id,
        "message": request.message,
    }

    for layer in PIPELINE:
        pipeline_state = layer.process(pipeline_state)

    return pipeline_state


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
