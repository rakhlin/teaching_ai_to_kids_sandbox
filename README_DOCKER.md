# Alternating Word Chat - Docker Package

This Docker image contains the Alternating Word Chat application with pre-downloaded AI models.

## Prerequisites

- Docker installed on your system
- ~4GB disk space for the Docker image (includes both models)

## Quick Start

### Option 1: Build and Run

```bash
# Build the Docker image (this will download models - takes 5-10 minutes)
./build-and-run.sh

# Run the application
docker run -it alternating-word-chat:latest
```

### Option 2: Using Docker Compose

```bash
# Build and run with docker-compose
docker-compose up --build

# Or run in background
docker-compose up -d
```

## Running with Options

```bash
# With custom temperature
docker run -it alternating-word-chat:latest --temperature 0.7

# With word frequency histogram (100 samples)
docker run -it alternating-word-chat:latest --hist 100

# With limited display and context
docker run -it alternating-word-chat:latest --display-words 20 --context-words 50
```

## Available Command-Line Arguments

- `--temperature <float>`: Set AI creativity (0.0-2.0, default: 0.2)
- `--hist <int>`: Enable word frequency analysis (0=disabled, default: 0)
- `--display-words <int>`: Limit displayed words (default: all)
- `--context-words <int>`: Limit AI context (default: all)
- `--prediction-time <float>`: Seconds to show prediction (default: 0)

## In-Game Commands

Once running, you can use these commands:
- `/help` - Show help menu
- `/model qwen` or `/model tinystories` - Switch between models
- `/temp <0.0-2.0>` - Adjust temperature
- `/hist <0-1000>` - Set frequency analysis samples
- `/restart` - Start a new story
- `/quit` - Exit the application

## Notes

- Voice input features are disabled in Docker (requires local audio access)
- The first build will take 5-10 minutes to download models (~3GB)
- Subsequent runs will be instant as models are cached in the image

## Sharing the Image

Once built, you can save and share the Docker image:

```bash
# Save the image to a tar file
docker save alternating-word-chat:latest | gzip > alternating-word-chat.tar.gz

# Load on another machine
docker load < alternating-word-chat.tar.gz
```

## Troubleshooting

If you encounter memory issues, increase Docker's memory allocation in Docker Desktop settings (recommend at least 4GB).