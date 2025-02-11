#!/bin/bash

# URL of the FastAPI endpoint
API_URL="http://localhost:8000/model/unload"

# Send the DELETE request to the API
curl -X DELETE "$API_URL"