#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t alternating-word-chat:latest .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "To run the application:"
    echo "  docker run -it alternating-word-chat:latest"
    echo ""
    echo "Or with command-line arguments:"
    echo "  docker run -it alternating-word-chat:latest --temperature 0.7 --hist 100"
else
    echo "Build failed!"
    exit 1
fi