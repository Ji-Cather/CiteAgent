#!/bin/bash

# test_vllm.sh
# Test if vllm api_server is running and responding

API_KEY="sk-api"
HOST="127.0.0.1"
PORT="8001"

# Wait for server to start (adjust sleep if needed)
sleep 5

# Test: Check if server is up
curl -s "http://${HOST}:${PORT}/v1/models" -H "Authorization: Bearer ${API_KEY}" | grep -q "llama3_8b"
if [ $? -eq 0 ]; then
    echo "✅ vllm server is running and model is loaded."
else
    echo "❌ vllm server test failed."
    exit 1
fi

# Test: Send a simple completion request
RESPONSE=$(curl -s "http://${HOST}:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d '{"model":"llama3_8b","prompt":"Hello, world!","max_tokens":5}')

echo "$RESPONSE"
# if [ $? -eq 0 ]; then
#     echo "✅ Completion API responded successfully."
# else
#     echo "❌ Completion API test failed."
#     exit 1
# fi