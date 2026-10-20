# GPA Predictor

Predicts college GPA from Math and Verbal SAT scores. Trained with scikit-learn,
served with FastAPI, packaged with Docker.

```
data/sat_gpa.csv  ->  train.py  ->  model/model.joblib  ->  app/main.py (FastAPI)  ->  Docker  ->  cloud
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt

python train.py                 # trains models, saves the best one to model/
uvicorn app.main:app --reload   # http://localhost:8000
pytest                          # run tests
```

- `http://localhost:8000/` web form
- `http://localhost:8000/docs` interactive API docs (Swagger)
- `POST /predict` `{"math_sat": 650, "verbal_sat": 600}` -> `{"predicted_gpa": 3.33, ...}`

## Run with Docker

```bash
docker build -t gpa-predictor .
docker run -p 8000:8000 gpa-predictor
```

## Deploy (Render, free tier)

1. Push this folder to a GitHub repo.
2. On https://render.com: New -> Web Service -> pick the repo.
3. Render detects the `Dockerfile` automatically. Click Deploy.

## Results

Three models are compared with 5-fold cross-validation (linear regression,
polynomial ridge, random forest). See `model/metrics.json`. SAT scores explain
roughly 40% of GPA variance (R^2 ~ 0.4), with an average error of ~0.3 GPA points.
