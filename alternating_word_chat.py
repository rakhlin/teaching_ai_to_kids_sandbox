#!/usr/bin/env python3

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import warnings
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    import speech_recognition as sr
    HAS_SPEECH_RECOGNITION = True
except ImportError:
    HAS_SPEECH_RECOGNITION = False
import subprocess
import re
import sys
import time
import threading
from typing import Optional, Tuple, List
from collections import Counter

class AlternatingWordChat:
    def __init__(self, use_voice=False, speak_mode="full", prediction_display_time=0, display_words=None, context_words=None, temperature=0.2, hist_samples=0):
        print("Loading models...")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.speak_mode = speak_mode  # "full" or "last"
        self.prediction_display_time = prediction_display_time  # seconds to show prediction
        self.display_words = display_words  # None = all, or number of recent words to show
        self.context_words = context_words  # None = all, or number of words for model context
        self.temperature = temperature  # Temperature for text generation (0.0-2.0)
        self.hist_samples = hist_samples  # Number of samples for frequency histogram (0 = disabled)
        self.solo_mode = False  # Solo mode: only human words added to context
        
        # Load both models
        print("Loading Qwen2.5-1.5B model...")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
            device_map="auto"
        )
        
        print("Loading TinyStories-33M model...")
        self.tinystories_tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
        # Add padding token if it doesn't exist
        if self.tinystories_tokenizer.pad_token is None:
            self.tinystories_tokenizer.pad_token = self.tinystories_tokenizer.eos_token
        self.tinystories_model = AutoModelForCausalLM.from_pretrained(
            "roneneldan/TinyStories-33M",
            torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
            device_map="auto"
        )
        
        # Set default model to Qwen
        self.current_model_name = "qwen"
        self.tokenizer = self.qwen_tokenizer
        self.model = self.qwen_model
        
        self.use_voice = use_voice

        if self.use_voice:
            if not HAS_SPEECH_RECOGNITION:
                print("Voice input requires SpeechRecognition and pyaudio.")
                print("Install them with: pip install SpeechRecognition pyaudio")
                print("(portaudio system library also required: brew install portaudio)")
                print("Falling back to text input.")
                self.use_voice = False
            else:
                self.recognizer = sr.Recognizer()
                self.recognizer.energy_threshold = 300
                self.recognizer.dynamic_energy_threshold = False
                self.recognizer.pause_threshold = 0.8
                self.microphone = sr.Microphone()

                with self.microphone as source:
                    print("Adjusting for ambient noise...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        self.conversation_history = []
        self.thinking_animation_stop = threading.Event()
        self.display_lock = threading.Lock()
        
    def show_thinking_animation(self):
        """Show animated thinking indicator"""
        frames = [
            "⠋ Thinking...",
            "⠙ Thinking...",
            "⠹ Thinking...",
            "⠸ Thinking...",
            "⠼ Thinking...",
            "⠴ Thinking...",
            "⠦ Thinking...",
            "⠧ Thinking...",
            "⠇ Thinking...",
            "⠏ Thinking..."
        ]
        
        frame_idx = 0
        while not self.thinking_animation_stop.is_set():
            with self.display_lock:
                self.move_cursor(13)
                self.clear_line()
                print(f"   \033[96m{frames[frame_idx]}\033[0m", end='', flush=True)
            frame_idx = (frame_idx + 1) % len(frames)
            time.sleep(0.1)
        
    def move_cursor(self, row, col=0):
        """Move cursor to specific position"""
        print(f"\033[{row};{col}H", end='')
    
    def clear_line(self):
        """Clear current line"""
        print("\033[2K", end='')
        
    def setup_display(self):
        """Setup initial display layout"""
        # Clear screen and move cursor to home position
        print("\033[2J\033[H", end='')
        sys.stdout.flush()
        
        # Header with fancy decorations
        print("╔" + "═"*68 + "╗")  # Line 1
        print("║" + " " * 17 + "✨ 🐭 MOUSELAND: CHAT WITH AI 🐭 ✨" + " " * 16 + "║")  # Line 2
        print("╚" + "═"*68 + "╝")  # Line 3
        
        # Reserve lines for display
        print("📚 【 Story so far 】")  # Line 4
        print()  # Line 5 - blank
        print("   " + " " * 65)  # Line 6 - for story
        print()  # Line 7 - blank
        print("•" + "·"*68 + "•")  # Line 8
        print()  # Line 9 - for input
        print()  # Line 10 - blank
        print("≈" * 35 + "🌊" + "≈" * 34)  # Line 11 - wavy separator
        model_display = "Qwen" if self.current_model_name == "qwen" else "TinyStories"
        print(f"🧠 ✦ Robot's Brain ({model_display}) ✦")  # Line 12
        print("   " + " " * 65)  # Line 13 - for prediction
        print("─" * 70)  # Line 14 - separator
        print()  # Line 15 - blank line
        print("─" * 70)  # Line 16 - horizontal divider
        print("\033[90m💡 /help /restart /model /words /context /temp /hist /speak /solo /quit\033[0m")  # Line 17 - help
        print(f"\033[90m🌡️ {self.temperature:.3f}  📊 {self.hist_samples}\033[0m")  # Line 18 - temp and hist display
        # Only show frequency lines if hist_samples > 0
        if self.hist_samples > 0:
            print("─" * 70)  # Line 19 - separator for word frequencies
            print(f"\033[90m🎲 Most likely next words (from {self.hist_samples} samples):\033[0m")  # Line 20
            print(" " * 70)  # Line 21 - placeholder for frequencies
        
    def update_display(self, continuation="", word_frequencies=None, other_continuation=""):
        """Update specific lines without clearing screen"""
        # Update story line (line 6)
        self.move_cursor(6)
        self.clear_line()
        
        # Apply word limit if specified
        if self.display_words is not None and len(self.conversation_history) > self.display_words:
            # Show only the last N words with ellipsis
            display_words = self.conversation_history[-self.display_words:]
            current_text = "... " + ' '.join(display_words)
        else:
            current_text = ' '.join(self.conversation_history)
        
        # Truncate if too long for display width
        if len(current_text) > 58:
            current_text = "..." + current_text[-55:]
        print(f"   \033[91m『 {current_text} 』\033[0m", end='')
        
        # Update prediction line (line 13) - at the bottom with nice formatting
        self.move_cursor(13)
        self.clear_line()
        if continuation:
            # Split by newlines and take only the first line
            first_line = continuation.split('\n')[0].strip()
            # Make prediction pretty with yellow text and fancy symbols
            cont_display = first_line[:50] if len(first_line) <= 50 else first_line[:47] + "..."
            # If we have two models' predictions, show both
            if other_continuation and self.current_model_name == "tinystories":
                print(f"   \033[93m🧸 Tiny: {cont_display}\033[0m", end='')
            else:
                print(f"   \033[93m🤖🧠 ✧ ⟨ {cont_display} ⟩ ✧\033[0m", end='')
        else:
            print("   ", end='')
        
        # Handle line 14 for the other model's prediction
        self.move_cursor(14)
        self.clear_line()
        if other_continuation and self.current_model_name == "tinystories":
            # Split by newlines and take only the first line
            qwen_first_line = other_continuation.split('\n')[0].strip()
            qwen_display = qwen_first_line[:50] if len(qwen_first_line) <= 50 else qwen_first_line[:47] + "..."
            print(f"   \033[94m🤖 Qwen: {qwen_display}\033[0m", end='')
        else:
            # Clear the line but don't print anything
            print("", end='')
        
        # Update word frequencies (line 21) - only if hist_samples > 0
        if word_frequencies and self.hist_samples > 0:
            self.move_cursor(21)
            self.clear_line()
            # Format the top 10 words with their frequencies
            freq_display = "  "
            for word, count in word_frequencies[:10]:
                freq_display += f"{word}({count}) "
            # Truncate if too long
            if len(freq_display) > 68:
                freq_display = freq_display[:65] + "..."
            print(f"\033[90m{freq_display}\033[0m", end='')
        
        # Move cursor to input line (line 9)
        self.move_cursor(9)
        sys.stdout.flush()
        
    def speak_text(self, text):
        """Use macOS 'say' command with Alex voice for text-to-speech"""
        try:
            subprocess.run(["say", "-v", "Alex", text], check=False)
        except Exception as e:
            pass  # Silent fail for TTS
        
    def get_initial_prompt(self) -> str:
        self.setup_display()
        self.update_display()
        
        while True:
            # Make sure we're at line 9 for input
            self.move_cursor(9)
            self.clear_line()
            prompt = input("🖊️  Your initial prompt: ").strip()
            # Clear the input line after reading
            self.move_cursor(9)
            self.clear_line()
            
            # Check if it's a command
            if prompt.startswith('/'):
                # Process command using get_text_input logic
                result = self.process_command(prompt)
                if result == 'RESTART':
                    return 'RESTART'  # Signal restart
                elif result is None:
                    return None  # Signal quit
                # Otherwise continue asking for initial prompt
                # Note: process_command already positions cursor at line 10
                continue
            elif prompt:
                # Valid prompt, pronounce and return it
                self.speak_text(prompt)
                return prompt
            # Empty prompt, ask again
    
    def extract_first_word(self, text: str, context: str) -> Optional[str]:
        """Extract first word from generated text, returns None if no valid word found"""
        # Skip leading punctuation and whitespace to find the first word
        search_text = text.strip().lstrip('.,!?;:\'"— \n\t')

        # Match first word with optional trailing punctuation (no underscores)
        word_pattern = re.match(r'^([a-zA-Z0-9]+[.,!?;:]?)', search_text)

        if word_pattern:
            word = word_pattern.group(1)  # Keep original capitalization
            context_words = context.split()
            # Check if it's a repetition (compare without punctuation and case)
            if context_words:
                last_word_clean = re.sub(r'[.,!?;:]', '', context_words[-1].lower())
                word_clean = re.sub(r'[.,!?;:]', '', word.lower())
                if word_clean == last_word_clean:
                    # Try to get the next word
                    remaining = search_text[len(word_pattern.group(1)):].strip()
                    remaining = remaining.lstrip('.,!?;:\'"— \n\t')
                    next_word_pattern = re.match(r'^([a-zA-Z0-9]+[.,!?;:]?)', remaining)
                    if next_word_pattern:
                        return next_word_pattern.group(1)
                    else:
                        return None  # No valid non-repeating word found
            return word
        return None  # No valid word found
    
    def generate_single_word(self, context: str) -> Tuple[str, str, List, str]:
        """Generate single word and return (word, full_continuation, word_frequencies, other_model_continuation)"""
        prompt = context
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Start thinking animation in separate thread
        self.thinking_animation_stop.clear()
        animation_thread = threading.Thread(target=self.show_thinking_animation)
        animation_thread.start()
        
        # Generate 100 samples to get word frequencies (using temp=1.0 for diversity)
        word_counter = Counter()
        main_continuation = None
        other_model_continuation = None
        
        with torch.no_grad():
            # First generate with user's temperature for the actual word
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=20,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            new_tokens = outputs[0][len(inputs.input_ids[0]):]
            main_continuation = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            # If TinyStories is selected, also generate from Qwen
            if self.current_model_name == "tinystories":
                qwen_inputs = self.qwen_tokenizer(prompt, return_tensors="pt").to(self.device)
                qwen_outputs = self.qwen_model.generate(
                    qwen_inputs.input_ids,
                    attention_mask=qwen_inputs.attention_mask,
                    max_new_tokens=20,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.qwen_tokenizer.eos_token_id,
                    eos_token_id=self.qwen_tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
                qwen_new_tokens = qwen_outputs[0][len(qwen_inputs.input_ids[0]):]
                other_model_continuation = self.qwen_tokenizer.decode(qwen_new_tokens, skip_special_tokens=True).strip()
            
            # Now generate samples with temperature=1.0 for frequency analysis (if enabled)
            if self.hist_samples > 0:
                for i in range(self.hist_samples):
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=20,
                        temperature=1.0,  # Fixed temperature for diversity
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
                    
                    new_tokens = outputs[0][len(inputs.input_ids[0]):]
                    continuation = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
                    # Extract first word and count it (skip degenerate samples)
                    first_word = self.extract_first_word(continuation, context)
                    if first_word is not None:
                        # Strip trailing punctuation so "milk" and "milk," count the same
                        word_counter[re.sub(r'[.,!?;:]$', '', first_word.lower())] += 1
        
        # Stop animation
        self.thinking_animation_stop.set()
        animation_thread.join()
        
        # Clear animation line
        self.move_cursor(13)
        self.clear_line()
        
        # Get the most common words (only if we collected samples)
        most_common = word_counter.most_common(10) if self.hist_samples > 0 else []
        
        # Extract the actual word to use (from the first/main continuation)
        word = self.extract_first_word(main_continuation, context)
        if word is None:
            word = "the"  # Fallback only for the actual played word, not histogram

        return word, main_continuation, most_common, other_model_continuation
    
    def process_command(self, word: str) -> Optional[str]:
        """Process commands - returns None for quit, 'RESTART' for restart, 'CONTINUE' for continue"""
        if word.lower() == '/help':
            self.show_help_overlay()
            return 'CONTINUE'
        elif word.lower() in ['/quit', '/exit']:
            return None
        elif word.lower() == '/restart':
            return 'RESTART'
        elif word.lower().startswith('/words'):
            # Change display words setting
            parts = word.split()
            if len(parts) > 1:
                try:
                    num = parts[1]
                    if num.lower() == 'all':
                        self.display_words = None
                    else:
                        self.display_words = int(num)
                    self.update_display()
                    # Show confirmation on line 15
                    self.move_cursor(14)
                    self.clear_line()
                    print(f"  ✅ Display set to {'all' if self.display_words is None else f'{self.display_words} recent'} words", end='', flush=True)
                    time.sleep(1.5)
                    self.move_cursor(14)
                    self.clear_line()
                    self.move_cursor(9)
                except (IndexError, ValueError):
                    # Show error on line 15
                    self.move_cursor(14)
                    self.clear_line()
                    print("  ❌ Usage: /words <number> or /words all", end='', flush=True)
                    time.sleep(1.5)
                    self.move_cursor(14)
                    self.clear_line()
                    self.move_cursor(9)
            return 'CONTINUE'
        elif word.lower().startswith('/context'):
            # Change context words for model
            parts = word.split()
            if len(parts) > 1:
                try:
                    num = parts[1]
                    if num.lower() == 'all':
                        self.context_words = None
                    else:
                        self.context_words = int(num)
                    # Show confirmation on line 15
                    self.move_cursor(14)
                    self.clear_line()
                    print(f"  ✅ Context set to {'all' if self.context_words is None else f'{self.context_words} recent'} words for AI", end='', flush=True)
                    time.sleep(1.5)
                    self.move_cursor(14)
                    self.clear_line()
                    self.move_cursor(9)
                except (IndexError, ValueError):
                    # Show error on line 15
                    self.move_cursor(14)
                    self.clear_line()
                    print("  ❌ Usage: /context <number> or /context all", end='', flush=True)
                    time.sleep(1.5)
                    self.move_cursor(14)
                    self.clear_line()
                    self.move_cursor(9)
            return 'CONTINUE'
        elif word.lower().startswith('/hist'):
            # Change histogram samples setting
            parts = word.split()
            if len(parts) > 1:
                try:
                    samples = int(parts[1])
                    if samples < 0:
                        samples = 0
                    elif samples > 1000:
                        samples = 1000  # Cap at 1000 for performance
                    
                    self.hist_samples = samples
                    # Update the temperature/hist display
                    self.update_temperature_display()
                    # Show confirmation on line 15
                    self.move_cursor(14)
                    self.clear_line()
                    if samples == 0:
                        print(f"  ✅ Word frequency analysis disabled", end='', flush=True)
                    else:
                        print(f"  ✅ Word frequency analysis: {self.hist_samples} samples", end='', flush=True)
                    time.sleep(1.5)
                    self.move_cursor(14)
                    self.clear_line()
                    # Re-setup display to show/hide frequency lines
                    self.setup_display()
                    self.update_display()
                except ValueError:
                    # Show error on line 15
                    self.move_cursor(14)
                    self.clear_line()
                    print("  ❌ Usage: /hist <number> (0-1000, 0=disabled)", end='', flush=True)
                    time.sleep(1.5)
                    self.move_cursor(14)
                    self.clear_line()
            else:
                # Show current value
                self.move_cursor(14)
                self.clear_line()
                if self.hist_samples == 0:
                    print(f"  🔄 Word frequency analysis is disabled", end='', flush=True)
                else:
                    print(f"  🔄 Current: {self.hist_samples} samples", end='', flush=True)
                time.sleep(1.5)
                self.move_cursor(14)
                self.clear_line()
            self.move_cursor(9)
            return 'CONTINUE'
        elif word.lower().startswith('/speak'):
            # Change speak mode
            parts = word.split()
            if len(parts) > 1:
                mode = parts[1].lower()
                if mode == 'last':
                    self.speak_mode = 'last'
                    # Show confirmation on line 15
                    self.move_cursor(14)
                    self.clear_line()
                    print(f"  ✅ Speech mode: speaking last word only", end='', flush=True)
                    time.sleep(1.5)
                    self.move_cursor(14)
                    self.clear_line()
                    self.move_cursor(9)
                elif mode == 'all' or mode == 'full':
                    self.speak_mode = 'full'
                    # Show confirmation on line 15
                    self.move_cursor(14)
                    self.clear_line()
                    print(f"  ✅ Speech mode: speaking full sentence", end='', flush=True)
                    time.sleep(1.5)
                    self.move_cursor(14)
                    self.clear_line()
                    self.move_cursor(9)
                else:
                    # Show error on line 15
                    self.move_cursor(14)
                    self.clear_line()
                    print("  ❌ Usage: /speak last or /speak all", end='', flush=True)
                    time.sleep(1.5)
                    self.move_cursor(14)
                    self.clear_line()
                    self.move_cursor(9)
            else:
                # Toggle between modes if no argument given
                self.speak_mode = 'last' if self.speak_mode == 'full' else 'full'
                # Show confirmation on line 15
                self.move_cursor(14)
                self.clear_line()
                print(f"  ✅ Speech mode: speaking {'last word only' if self.speak_mode == 'last' else 'full sentence'}", end='', flush=True)
                time.sleep(1.5)
                self.move_cursor(14)
                self.clear_line()
                self.move_cursor(9)
            return 'CONTINUE'
        elif word.lower().startswith('/solo'):
            # Toggle or set solo mode
            parts = word.split()
            if len(parts) > 1:
                mode = parts[1].lower()
                if mode == 'on':
                    self.solo_mode = True
                    # Show confirmation on line 15
                    self.move_cursor(14)
                    self.clear_line()
                    print(f"  ✅ Solo mode ON: Only human words added to context", end='', flush=True)
                    time.sleep(1.5)
                    self.move_cursor(14)
                    self.clear_line()
                    self.move_cursor(9)
                elif mode == 'off':
                    self.solo_mode = False
                    # Show confirmation on line 15
                    self.move_cursor(14)
                    self.clear_line()
                    print(f"  ✅ Solo mode OFF: Alternating words with AI", end='', flush=True)
                    time.sleep(1.5)
                    self.move_cursor(14)
                    self.clear_line()
                    self.move_cursor(9)
                else:
                    # Show error on line 15
                    self.move_cursor(14)
                    self.clear_line()
                    print("  ❌ Usage: /solo on or /solo off", end='', flush=True)
                    time.sleep(1.5)
                    self.move_cursor(14)
                    self.clear_line()
                    self.move_cursor(9)
            else:
                # Toggle mode if no argument given
                self.solo_mode = not self.solo_mode
                # Show confirmation on line 15
                self.move_cursor(14)
                self.clear_line()
                status = "ON: Only human words added" if self.solo_mode else "OFF: Alternating with AI"
                print(f"  ✅ Solo mode {status}", end='', flush=True)
                time.sleep(1.5)
                self.move_cursor(14)
                self.clear_line()
                self.move_cursor(9)
            return 'CONTINUE'
        elif word.lower().startswith('/temp'):
            # Change temperature setting
            parts = word.split()
            if len(parts) > 1:
                try:
                    temp = float(parts[1])
                    if temp <= 0:
                        temp = 0.001  # Use 0.001 for zero or negative values
                    elif temp > 2.0:
                        temp = 2.0  # Cap at 2.0
                    
                    self.temperature = temp
                    # Update temperature/hist display
                    self.update_temperature_display()
                    # Show confirmation on line 15
                    self.move_cursor(14)
                    self.clear_line()
                    print(f"  ✅ Temperature set to {self.temperature:.3f}", end='', flush=True)
                    time.sleep(1.5)
                    self.move_cursor(14)
                    self.clear_line()
                    self.move_cursor(9)
                except (IndexError, ValueError):
                    # Show error on line 15
                    self.move_cursor(14)
                    self.clear_line()
                    print("  ❌ Usage: /temp <number> (0.0-2.0)", end='', flush=True)
                    time.sleep(1.5)
                    self.move_cursor(14)
                    self.clear_line()
                    self.move_cursor(9)
            return 'CONTINUE'
        elif word.lower().startswith('/model'):
            # Switch between models
            parts = word.split()
            if len(parts) > 1:
                model_name = parts[1].lower()
                if model_name in ['qwen', 'tinystories', 'tiny']:
                    if model_name in ['tinystories', 'tiny']:
                        model_name = 'tinystories'
                    
                    # Switch model
                    self.current_model_name = model_name
                    if model_name == 'qwen':
                        self.tokenizer = self.qwen_tokenizer
                        self.model = self.qwen_model
                        display_name = "Qwen2.5-1.5B"
                    else:
                        self.tokenizer = self.tinystories_tokenizer
                        self.model = self.tinystories_model
                        display_name = "TinyStories-33M"
                    
                    # Update the model display
                    self.update_model_display()
                    
                    # Show confirmation on line 15
                    self.move_cursor(14)
                    self.clear_line()
                    print(f"  ✅ Switched to {display_name} model", end='', flush=True)
                    time.sleep(1.5)
                    self.move_cursor(14)
                    self.clear_line()
                    self.move_cursor(9)
                else:
                    # Show error on line 15
                    self.move_cursor(14)
                    self.clear_line()
                    print("  ❌ Usage: /model qwen or /model tinystories", end='', flush=True)
                    time.sleep(1.5)
                    self.move_cursor(14)
                    self.clear_line()
                    self.move_cursor(9)
            else:
                # Show current model
                display_name = "Qwen2.5-1.5B" if self.current_model_name == "qwen" else "TinyStories-33M"
                self.move_cursor(14)
                self.clear_line()
                print(f"  🤖 Current model: {display_name}", end='', flush=True)
                time.sleep(1.5)
                self.move_cursor(14)
                self.clear_line()
                self.move_cursor(9)
            return 'CONTINUE'
        else:
            # Unknown command
            self.move_cursor(14)
            self.clear_line()
            print("  ❌ Unknown command. Type a word or use /help", end='', flush=True)
            time.sleep(1.5)
            self.move_cursor(14)
            self.clear_line()
            self.move_cursor(9)
            return 'CONTINUE'
    
    def get_text_input(self) -> Optional[str]:
        while True:
            # Ensure line is clear before prompting
            self.move_cursor(9)
            self.clear_line()
            word = input("👤 Your word: ").strip()

            # Clear the input line after reading
            self.move_cursor(9)
            self.clear_line()

            # Check if it's a command
            if word.startswith('/'):
                result = self.process_command(word)
                if result == 'CONTINUE':
                    continue  # Get next input
                else:
                    return result  # None or 'RESTART'

            # Not a command, process as regular word
            # Allow punctuation attached to words (e.g., "dog.", "cat,", "run!")
            # Extract first word with optional punctuation
            match = re.match(r'^([a-zA-Z0-9]+[.,!?;:]?)', word.strip())

            if match:
                selected_word = match.group(1)
                # Check if there are extra words after
                remaining = word[len(selected_word):].strip()
                if remaining:
                    self.update_display()
                    time.sleep(1)
                return selected_word
            else:
                self.update_display()
                time.sleep(1)
                continue  # Ask again
    
    def get_voice_input(self) -> Optional[str]:
        self.update_display()
        
        # Show listening prompt with pretty formatting
        self.move_cursor(9)
        self.clear_line()
        print("🎤 ✧ Listening... ✧", end='', flush=True)
        
        # Play beep
        subprocess.run(["afplay", "/System/Library/Sounds/Tink.aiff"], check=False)
        
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=2)
                
            try:
                result = self.recognizer.recognize_google(audio, language="en-US", show_all=True)
                if result and 'alternative' in result:
                    text = result['alternative'][0]['transcript']
                else:
                    text = self.recognizer.recognize_google(audio, language="en-US")
                
                if text:
                    text = text.strip().lower()
                    words = text.split()
                    
                    if words:
                        word = words[0]
                        # Clear the listening prompt
                        self.move_cursor(9)
                        self.clear_line()
                        self.update_display()
                        time.sleep(1)
                        return word
                    
            except sr.UnknownValueError:
                # Clear the listening prompt
                self.move_cursor(9)
                self.clear_line()
                self.update_display()
                time.sleep(1)
                return None
            except Exception as e:
                # Clear the listening prompt
                self.move_cursor(9)
                self.clear_line()
                self.update_display()
                time.sleep(1)
                return None
                
        except sr.WaitTimeoutError:
            # Clear the listening prompt
            self.move_cursor(9)
            self.clear_line()
            self.update_display()
            time.sleep(1)
            return None
    
    def update_help(self):
        """Update help information at the bottom of the screen"""
        self.move_cursor(17)
        self.clear_line()
        print("\033[90m💡 /help /restart /model /words /context /temp /hist /speak /solo /quit\033[0m", end='', flush=True)
        self.move_cursor(9)
    
    def update_temperature_display(self):
        """Update temperature and hist display on line 18"""
        self.move_cursor(18)
        self.clear_line()
        print(f"\033[90m🌡️ {self.temperature:.3f}  📊 {self.hist_samples}\033[0m", end='', flush=True)
        self.move_cursor(9)
    
    def update_model_display(self):
        """Update model name display on line 12"""
        self.move_cursor(12)
        self.clear_line()
        model_display = "Qwen" if self.current_model_name == "qwen" else "TinyStories"
        print(f"🧠 ✦ Robot's Brain ({model_display}) ✦", end='', flush=True)
        self.move_cursor(9)
    
    def show_help_overlay(self):
        """Display help overlay window with all commands and options"""
        # Save cursor position
        print("\033[s", end='')
        
        # Clear screen and show help window
        print("\033[2J\033[H", end='')
        
        # Draw help window
        print("\n" + "╔" + "═"*68 + "╗")
        print("║" + " " * 27 + "📚 HELP MENU 📚" + " " * 26 + "║")
        print("╠" + "═"*68 + "╣")
        
        # Commands
        print("║ " + "\033[96mCOMMANDS:\033[0m" + " " * 57 + "║")
        print("║" + " " * 68 + "║")
        print("║  \033[93m/help\033[0m        - Show this help menu" + " " * 30 + "║")
        print("║  \033[93m/quit\033[0m, \033[93m/exit\033[0m - Exit the program" + " " * 33 + "║")
        print("║  \033[93m/restart\033[0m     - Restart the story" + " " * 32 + "║")
        print("║" + " " * 68 + "║")
        
        # Model selection
        print("║ " + "\033[96mMODEL SELECTION:\033[0m" + " " * 50 + "║")
        print("║  \033[93m/model\033[0m       - Show current model" + " " * 31 + "║")
        print("║  \033[93m/model qwen\033[0m  - Switch to Qwen2.5-1.5B (sophisticated)" + " " * 10 + "║")
        print("║  \033[93m/model tinystories\033[0m or \033[93m/model tiny\033[0m - Switch to TinyStories-33M" + " " * 1 + "║")
        print("║" + " " * 68 + "║")
        
        # Display settings
        print("║ " + "\033[96mDISPLAY SETTINGS:\033[0m" + " " * 49 + "║")
        print("║  \033[93m/words\033[0m       - Show current display word limit" + " " * 17 + "║")
        print("║  \033[93m/words <n>\033[0m   - Show only last n words (e.g., /words 10)" + " " * 8 + "║")
        print("║  \033[93m/words all\033[0m   - Show all words" + " " * 35 + "║")
        print("║" + " " * 68 + "║")
        
        # AI settings
        print("║ " + "\033[96mAI SETTINGS:\033[0m" + " " * 54 + "║")
        print("║  \033[93m/context\033[0m     - Show current context word limit" + " " * 17 + "║")
        print("║  \033[93m/context <n>\033[0m - Use only last n words for AI (e.g., /context 50)" + " " * 0 + "║")
        print("║  \033[93m/context all\033[0m - Use all words for AI context" + " " * 20 + "║")
        print("║" + " " * 68 + "║")
        print("║  \033[93m/temp\033[0m        - Show current temperature" + " " * 25 + "║")
        print("║  \033[93m/temp <n>\033[0m    - Set temperature (0.0-2.0, e.g., /temp 0.5)" + " " * 6 + "║")
        print("║                Lower = more predictable, Higher = more creative" + " " * 2 + "║")
        print("║" + " " * 68 + "║")
        print("║  \033[93m/hist\033[0m        - Show current histogram samples" + " " * 18 + "║")
        print("║  \033[93m/hist <n>\033[0m    - Set word frequency samples (0-1000)" + " " * 13 + "║")
        print("║                0 = disabled, higher = more accurate predictions" + " " * 2 + "║")
        print("║" + " " * 68 + "║")
        
        # Speech settings
        print("║ " + "\033[96mSPEECH SETTINGS:\033[0m" + " " * 50 + "║")
        print("║  \033[93m/speak\033[0m       - Toggle speech mode" + " " * 31 + "║")
        print("║  \033[93m/speak last\033[0m  - Speak only the last word added" + " " * 17 + "║")
        print("║  \033[93m/speak all\033[0m   - Speak the full sentence" + " " * 25 + "║")
        print("║" + " " * 68 + "║")
        
        # Solo mode
        print("║ " + "\033[96mSOLO MODE:\033[0m" + " " * 56 + "║")
        print("║  \033[93m/solo\033[0m        - Toggle solo mode (human words only)" + " " * 13 + "║")
        print("║  \033[93m/solo on\033[0m     - Enable solo mode: AI shows predictions but" + " " * 6 + "║")
        print("║                doesn't add words to the story" + " " * 20 + "║")
        print("║  \033[93m/solo off\033[0m    - Disable solo mode: back to alternating" + " " * 9 + "║")
        print("║" + " " * 68 + "║")
        
        # Current settings
        print("╠" + "═"*68 + "╣")
        print("║ " + "\033[96mCURRENT SETTINGS:\033[0m" + " " * 49 + "║")
        model_name = "Qwen2.5-1.5B" if self.current_model_name == "qwen" else "TinyStories-33M"
        print(f"║  Model: {model_name:<20} Temperature: {self.temperature:.3f}" + " " * 18 + "║")
        display_str = "all" if self.display_words is None else str(self.display_words)
        context_str = "all" if self.context_words is None else str(self.context_words)
        print(f"║  Display: {display_str:<10} Context: {context_str:<10} Histogram: {self.hist_samples:<4}" + " " * 13 + "║")
        speak_str = "last word" if self.speak_mode == "last" else "full sentence"
        solo_str = "ON" if self.solo_mode else "OFF"
        print(f"║  Speech: {speak_str:<20} Solo mode: {solo_str:<3}" + " " * 23 + "║")
        print("╚" + "═"*68 + "╝")
        
        print("\n" + " " * 20 + "\033[92mPress Enter to continue...\033[0m", end='', flush=True)
        input()
        
        # Restore display
        self.setup_display()
        self.update_display()
    
    def run(self):
        while True:  # Outer loop for restarting
            self.conversation_history = []  # Reset history
            initial_prompt = self.get_initial_prompt()
            
            # Check if user wants to quit
            if initial_prompt is None:
                return  # Exit the program
            
            # Check if this is a restart signal (shouldn't happen at initial prompt, but just in case)
            if initial_prompt == 'RESTART':
                continue  # Restart the game
            
            self.conversation_history.append(initial_prompt)
            
            # Help is already shown in setup_display, no need to show again
            
            restart_game = False
            
            try:
                while not restart_game:
                    # Apply context limit if specified
                    if self.context_words is not None and len(self.conversation_history) > self.context_words:
                        context_history = self.conversation_history[-self.context_words:]
                    else:
                        context_history = self.conversation_history
                    context = " ".join(context_history)
                    
                    # AI's turn
                    self.update_display()
                    ai_word, continuation, word_frequencies, other_continuation = self.generate_single_word(context)
                    
                    # In solo mode, don't add AI word to context
                    if not self.solo_mode:
                        self.conversation_history.append(ai_word)
                    
                    self.update_display(
                        continuation=continuation,
                        word_frequencies=word_frequencies,
                        other_continuation=other_continuation
                    )
                    
                    # In solo mode, skip speaking since AI word isn't added
                    if not self.solo_mode:
                        # Speak the word if needed
                        if self.speak_mode == "last":
                            self.speak_text(ai_word)
                        else:
                            self.speak_text(' '.join(self.conversation_history))

                    # Wait if prediction display time is set
                    if self.prediction_display_time > 0:
                        time.sleep(self.prediction_display_time)
                    
                    # Human's turn - show prompt immediately after AI's word
                    if self.use_voice:
                        human_word = None
                        while human_word is None:
                            human_word = self.get_voice_input()
                            if human_word is None:
                                self.move_cursor(9)
                                self.clear_line()
                                retry = input("🔄 ⟨ Press Enter to retry, type '/quit' or '/restart' ⟩: ")
                                # Clear the input line after reading
                                self.move_cursor(9)
                                self.clear_line()
                                if retry.lower() in ['/quit', '/exit']:
                                    return  # Exit program
                                elif retry.lower() == '/restart':
                                    restart_game = True
                                    break  # Break to restart
                    else:
                        human_word = self.get_text_input()
                    
                    if human_word is None:
                        return  # Exit the program
                    elif human_word == 'RESTART':
                        restart_game = True
                        break  # Break inner loop to restart
                        
                    self.conversation_history.append(human_word)
                    
                    # Clear prediction only after human word is added (if not already cleared)
                    if self.prediction_display_time == 0:
                        self.update_display(
                            continuation=""  # Clear prediction after human's turn
                        )
                    else:
                        # Just update to show new word without prediction
                        self.update_display()
                    
                    # Speak the word if needed
                    if self.speak_mode == "last":
                        self.speak_text(human_word)
                    else:
                        self.speak_text(' '.join(self.conversation_history))
                    
                    time.sleep(1)
                
            except KeyboardInterrupt:
                return  # Exit on Ctrl+C
            
            if not restart_game:
                break  # Exit outer loop if not restarting
        
        # Final display (only shown when exiting, not restarting)
        if not restart_game:
            os.system('clear' if os.name == 'posix' else 'cls')
            print("\n" + "╔" + "═"*68 + "╗")
            print("║" + " " * 21 + "📚 ✨ FINAL STORY ✨ 📚" + " " * 22 + "║")
            print("╚" + "═"*68 + "╝")
            print("\n" + "   🌟 " + ' '.join(self.conversation_history) + " 🌟")
            print("\n" + "•" + "·"*68 + "•")
            print("\n   Thanks for playing! 🎭 ✨ 👋\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Alternating Word Chat with Qwen2.5-1.5B')
    parser.add_argument('--voice', action='store_true', help='Use voice input instead of text')
    parser.add_argument('--speak-last', action='store_true', help='Only speak the last word added')
    parser.add_argument('--prediction-time', type=float, default=0, help='Seconds to display model prediction (default: 0 = keep visible during input)')
    parser.add_argument('--display-words', type=int, default=None, help='Number of recent words to display (default: all)')
    parser.add_argument('--context-words', type=int, default=None, help='Number of recent words for AI context (default: all)')
    parser.add_argument('--temperature', type=float, default=0.2, help='Temperature for AI generation 0.0-2.0 (default: 0.2)')
    parser.add_argument('--hist', type=int, default=0, help='Number of samples for word frequency histogram (default: 0=disabled)')
    args = parser.parse_args()
    
    speak_mode = "last" if args.speak_last else "full"
    # Validate temperature (match /temp command behavior)
    temperature = args.temperature
    if temperature <= 0:
        temperature = 0.001
    elif temperature > 2.0:
        temperature = 2.0
    chat = AlternatingWordChat(use_voice=args.voice, speak_mode=speak_mode,
                              prediction_display_time=args.prediction_time,
                              display_words=args.display_words,
                              context_words=args.context_words,
                              temperature=temperature,
                              hist_samples=args.hist)
    chat.run()

if __name__ == "__main__":
    main()