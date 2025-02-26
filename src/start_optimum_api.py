import uvicorn
import logging
from api.optimum_api import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ov_api")

def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Start the OpenVINO Inference API server
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        reload: Whether to enable auto-reload on code changes
    """
    logger.info(f"Starting OpenVINO Inference API server on {host}:{port}")
    logger.info("Available endpoints:")
    logger.info("  - POST   /model/load      Load a model")
    logger.info("  - DELETE /model/unload    Unload current model")
    logger.info("  - POST   /generate/text   Generate text synchronously")
    logger.info("  - POST   /generate/stream Stream text generation")
    logger.info("  - GET    /status         Get model status")
    logger.info("  - GET    /docs           API documentation")
    
    # Start the server
    uvicorn.run(
        "api.optimum_api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    start_server() 