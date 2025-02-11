# OpenVINO Inference API Request Types

This document describes all available request types and their parameters for the OpenVINO Inference API.

## Model Management

### Load Model
- **Endpoint**: `POST /model/load`
- **Description**: Load a model with specified configuration
- **Request Body**:
  ```json
  {
    "load_config": {
      "model_id": "string",
      "use_cache": true,
      "device": "CPU",
      "export_model": false,
      "warmup_generation": true,
      "warmup_generation_num": 2
    },
    "ov_config": {
      "NUM_STREAMS": "",
      "PERFORMANCE_HINT": "",
      "PRECISION_HINT": ""
    }
  }
  ```
- **Example curl**:
  ```bash
  curl -X POST http://localhost:8000/model/load \
    -H "Content-Type: application/json" \
    -d '{
      "load_config": {
        "model_id": "/media/ecomm/c0889304-9e30-4f04-b290-c7db463872c6/Models/Pytorch/Ministral-3b-instruct-int4_asym-ov",
        "use_cache": true,
        "device": "CPU",
        "export_model": false,
        "warmup_generation": true,
        "warmup_generation_num": 2
      },
      "ov_config": {
        "PERFORMANCE_HINT": "None",
        "PRECISION_HINT": "None"
      }
    }'
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Model {model_id} loaded successfully"
  }
  ```

### Unload Model
- **Endpoint**: `DELETE /model/unload`
- **Description**: Unload the currently loaded model and free resources
- **Example curl**:
  ```bash
  curl -X DELETE http://localhost:8000/model/unload
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Model unloaded successfully"
  }
  ```

## Text Generation

### Generate Text (Synchronous)
- **Endpoint**: `POST /generate/text`
- **Description**: Generate text synchronously and return complete response
- **Request Body**:
  ```json
  {
    "prompt": "string",
    "max_new_tokens": 128,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "do_sample": true,
    "num_return_sequences": 1,
    "pad_token_id": null,
    "eos_token_id": "2"
  }
  ```
- **Example curl**:
  ```bash
  curl -X POST http://localhost:8000/generate/text \
    -H "Content-Type: application/json" \
    -d '{
      "prompt": "What is artificial intelligence?",
      "max_new_tokens": 128,
      "temperature": 0.7,
      "top_k": 50,
      "top_p": 0.9,
      "repetition_penalty": 1.1,
      "do_sample": true
    }'
  ```
- **Response**:
  ```json
  {
    "generated_text": "string",
    "performance_metrics": {
      "generation_time": float,
      "input_tokens": int,
      "output_tokens": int,
      "new_tokens": int,
      "eval_time": float,
      "tokens_per_second": float
    }
  }
  ```

### Generate Text (Streaming)
- **Endpoint**: `POST /generate/stream`
- **Description**: Generate text with streaming response using Server-Sent Events (SSE)
- **Request Body**: Same as synchronous generation
- **Example curl**:
  ```bash
  curl -X POST http://localhost:8000/generate/stream \
    -H "Content-Type: application/json" \
    -H "Accept: text/event-stream" \
    -d '{
      "prompt": "Write a short story about a robot.",
      "max_new_tokens": 256,
      "temperature": 0.8,
      "top_k": 50,
      "top_p": 0.9,
      "repetition_penalty": 1.1,
      "do_sample": true
    }'
  ```
- **Response**: Server-Sent Events stream
  ```
  data: <token_1>
  
  data: <token_2>
  
  ...
  
  data: [DONE]
  
  data: {performance_metrics}
  ```

## Status

### Get Model Status
- **Endpoint**: `GET /status`
- **Description**: Get current model status and configuration
- **Example curl**:
  ```bash
  curl http://localhost:8000/status
  ```
- **Response**:
  ```json
  {
    "status": "loaded",
    "model_id": "string",
    "device": "string"
  }
  ```
  or when no model is loaded:
  ```json
  {
    "status": "no_model",
    "model_id": null,
    "device": null
  }
  ```

## Error Responses

All endpoints may return the following error responses:

- **400 Bad Request**: When request parameters are invalid or no model is loaded
- **500 Internal Server Error**: When an error occurs during model operations
  ```json
  {
    "detail": "Error message"
  }
  ```
