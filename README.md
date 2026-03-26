# SafeLayer

Multi-layer AI safety analysis pipeline built with FastAPI.

## Architecture

SafeLayer processes every input through a **six-stage pipeline**, where each layer enriches a shared context dictionary before handing off to the next:

```
User Input
    │
    ▼
┌──────────────────────────┐
│  L0 – Normalizer         │  Cleans & standardises raw input
├──────────────────────────┤
│  L1 – Classifiers        │  Runs independent per-category safety classifiers
├──────────────────────────┤
│  L2 – PGM Reasoning      │  Correlates classifier signals via a probabilistic
│                          │  graphical model to detect multi-signal risk
├──────────────────────────┤
│  L3 – Session Scorer     │  Accumulates risk across the conversation session
├──────────────────────────┤
│  L4 – Explainability     │  Generates a human-readable explanation
├──────────────────────────┤
│  Response Engine          │  Maps the final risk score to a graduated action
│                          │  (allow / warn / block)
└──────────────────────────┘
    │
    ▼
 JSON Response
```

Each layer is a class with a `process(session_id, message, context) -> LayerResult` method. The shared `context` dict lets downstream layers read upstream outputs.

## Project Structure

```
safelayer/
├── main.py                      # FastAPI app – /analyze and /health endpoints
├── models/
│   └── schemas.py               # Pydantic request/response models
├── pipeline/
│   ├── __init__.py              # Re-exports all layer classes
│   ├── layer0_normalizer.py     # L0 – Input normalization
│   ├── layer1_classifiers.py    # L1 – Per-category classifiers
│   ├── layer2_pgm.py            # L2 – PGM correlation reasoning
│   ├── layer3_session.py        # L3 – Session risk scorer
│   ├── layer4_explainability.py # L4 – Explainability engine
│   └── response_engine.py       # Graduated response engine
├── benchmark/
│   └── test_suite.py            # Benchmark runner
├── frontend/                    # React app (scaffold – to be built)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
cd safelayer
pip install -r requirements.txt
python main.py
```

The server starts at **http://localhost:8000**.

### Endpoints

| Method | Path       | Description                        |
|--------|------------|------------------------------------|
| GET    | `/health`  | Health check – returns `{"status": "ok"}` |
| POST   | `/analyze` | Run the safety pipeline on a message |

### Example Request

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess-001", "message": "Hello world"}'
```

### Run Benchmarks

```bash
cd safelayer
python -m benchmark.test_suite
```

## Status

All pipeline layers are **stubs** returning dummy data. Implementation of real detection logic is the next step.
