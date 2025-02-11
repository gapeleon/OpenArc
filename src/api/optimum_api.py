from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional, AsyncIterator

from engine.optimum_inference_core import (
    OV_LoadModelConfig,
    OV_Config,
    OV_GenerationConfig,
    Optimum_InferenceCore,
)

app = FastAPI(title="OpenVINO Inference API")

# Global state to store model instance
model_instance: Optional[Optimum_InferenceCore] = None

@app.post("/model/load")
async def load_model(load_config: OV_LoadModelConfig, ov_config: OV_Config):
    """Load a model with the specified configuration"""
    global model_instance
    try:
        # Initialize new model
        model_instance = Optimum_InferenceCore(
            load_model_config=load_config,
            ov_config=ov_config,
        )
        
        # Load the model
        model_instance.load_model()
        
        return {"status": "success", "message": f"Model {load_config.id_model} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/model/unload")
async def unload_model():
    """Unload the current model"""
    global model_instance
    if model_instance:
        model_instance.util_unload_model()
        model_instance = None
        return {"status": "success", "message": "Model unloaded successfully"}
    return {"status": "success", "message": "No model was loaded"}

@app.post("/generate/text")
async def generate_text(generation_config: OV_GenerationConfig):
    """Generate text without streaming"""
    global model_instance
    if not model_instance:
        raise HTTPException(status_code=400, detail="No model loaded")

    try:
        generated_text, metrics = model_instance.generate_text(generation_config)
        return {
            "generated_text": generated_text,
            "performance_metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/stream")
async def generate_text_stream(generation_config: OV_GenerationConfig):
    """Generate a text stream asynchronously"""
    global model_instance
    if not model_instance:
        raise HTTPException(status_code=400, detail="No model loaded")

    async def text_stream() -> AsyncIterator[str]:
        async for token in model_instance.generate_stream(generation_config):
            yield token

    return StreamingResponse(text_stream(), media_type="text/event-stream")

@app.get("/status")
async def get_status():
    """Get current model status and performance metrics"""
    global model_instance
    if not model_instance:
        return {
            "status": "no_model",
            "id_model": None,
            "device": None
        }
    
    return {
        "status": "loaded",
        "id_model": model_instance.load_model_config.id_model,
        "device": model_instance.load_model_config.device
    }
