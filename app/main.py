from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import pandas as pd
from contextlib import asynccontextmanager

from src.inference import InferencePipeline

pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    try:
        pipeline = InferencePipeline()
    except Exception as e:
        print(f"Failed to initialize inference pipeline: {e}")
    yield


app = FastAPI(title="EPIAS Energy Forecast API", lifespan=lifespan)


class PredictionRequest(BaseModel):
    date: str = Field(default=None, description="YYYY-MM-DD format", examples=["2026-02-15"])


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": pipeline.model is not None if pipeline else False}


@app.post("/predict")
def predict(request: PredictionRequest):
    if pipeline is None or pipeline.model is None:
        raise HTTPException(status_code=503, detail="Model not initialized.")
    
    try:
        if request.date:
            target_date = pd.to_datetime(request.date)
        else:
            target_date = datetime.now() + timedelta(days=-1)
            
        results = pipeline.predict(target_date)
        
        return {
            "target_date": str(target_date.date()),
            "predictions": results.to_dict(orient="records")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
