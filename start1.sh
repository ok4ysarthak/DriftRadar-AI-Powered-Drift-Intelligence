#!/bin/sh

# This script pulls the model and then starts the Ollama server.
# It ensures the model is available before the API tries to use it.

echo "Pulling Llama 3 model..."
ollama pull llama3

echo "Starting Ollama server..."
ollama serve