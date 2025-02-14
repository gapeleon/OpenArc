#!/bin/bash

# URL of the FastAPI endpoint
API_URL="http://localhost:8000/generate/stream"

# JSON payload for the generation configuration
GENERATION_CONFIG='{
    "conversation": [
        {
            "role": "user",
            "content": "Hello!"
        }
    ],
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "do_sample": true,
    "num_return_sequences": 1,
    "pad_token_id": null,
    "eos_token_id": 2
}'

# Send the POST request to the API
curl -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -d "$GENERATION_CONFIG"
