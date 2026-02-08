# Alternating Word Chat

An interactive storytelling app where you and an AI take turns adding one word at a time to build a story together.

The app runs two language models locally on your Mac — no cloud API, no account needed:
- **Qwen2.5-1.5B-Instruct** — a 1.5 billion parameter model for sophisticated narratives
- **TinyStories-33M** — a lightweight 33 million parameter model for simple stories

You can switch between them at any time, adjust the AI's creativity (temperature), view word probability histograms, use voice input, and more.

## Installation (macOS)

There are two installation options:

| | `install.sh` | `install-no-sudo.sh` |
|---|---|---|
| **Requires admin/sudo** | Yes | No |
| **How it installs Python** | Via [Homebrew](https://brew.sh) | Via [Miniconda](https://docs.anaconda.com/miniconda/) (local to project folder) |
| **Voice input** (`--voice`) | Yes | No (requires system-level portaudio) |
| **Text-to-speech** (AI reads words aloud) | Yes | Yes |
| **Total download** | ~4 GB | ~4 GB |

Both installers download the same AI models (~3.5 GB) and create launcher scripts. The only functional difference is that the no-sudo version cannot use microphone voice input.

### Option 1: Standard install (requires sudo)

Installs Homebrew, Python 3, and portaudio system-wide. Supports all features including voice input.

```bash
git clone https://github.com/rakhlin/teaching_ai_to_kids_sandbox.git
cd teaching_ai_to_kids_sandbox
./install.sh
```

### Option 2: No-sudo install

Installs everything locally inside the project folder — no system-level changes, no password prompt. Use this if you don't have admin access on your Mac.

```bash
git clone https://github.com/rakhlin/teaching_ai_to_kids_sandbox.git
cd teaching_ai_to_kids_sandbox
./install-no-sudo.sh
```

Both options take about 10 minutes depending on your internet connection.

### Running

After installation, either:

- **Double-click** `start.command` in Finder
- **Or from Terminal:** `./start.sh`
- **With voice input:** `./start.sh --voice`

### Commands

Type these during the game:

| Command | Description |
|---|---|
| `/help` | Show all commands and current settings |
| `/model qwen` or `/model tiny` | Switch AI model |
| `/temp 0.5` | Set temperature (0.0–2.0, lower = more predictable) |
| `/hist 100` | Show word frequency analysis from N samples |
| `/words 10` | Show only last N words on screen |
| `/context 2` | Limit AI context to last N words |
| `/speak last` or `/speak all` | Speech output mode |
| `/solo on` | AI predicts but doesn't add words |
| `/restart` | Start a new story |
| `/quit` | Exit |

## Docker (alternative)

For Linux or if you prefer containers, see [README_DOCKER.md](README_DOCKER.md). Note: Docker mode loses macOS text-to-speech and MPS GPU acceleration.

## Room for improvement

There is a lot of room for improvement. Some ideas:

- **Better packaging** — a standalone `.app` bundle or Homebrew formula so users don't need to touch Terminal at all
- **Windows/Linux installer** — the current `install.sh` is macOS-only
- **Quantized models** — reduce the ~3.5 GB model download using 4-bit or 8-bit quantization
- **Web UI** — replace the terminal ANSI interface with a browser-based frontend
- **Longer context handling** — smarter summarization instead of simple truncation when the story gets long
- **Multi-word turns** — let the human or AI add phrases instead of single words
- **Save/load stories** — persist stories to disk and resume later
- **More models** — support additional models or let users bring their own
- **Streaming generation** — show the AI's word as it's being generated instead of waiting for the full output
