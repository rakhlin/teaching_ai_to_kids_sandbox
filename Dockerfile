# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Install system dependencies for audio (if needed)
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the models during build
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'); \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'); \
    AutoTokenizer.from_pretrained('roneneldan/TinyStories-33M'); \
    AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M')"

# Copy the application
COPY alternating_word_chat.py .

# Set environment variable to disable tokenizers parallelism warning
ENV TOKENIZERS_PARALLELISM=false

# Run the application
ENTRYPOINT ["python", "alternating_word_chat.py"]