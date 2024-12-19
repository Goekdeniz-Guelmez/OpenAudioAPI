# OpenAudioAPI

A versatile Speech API that supports both Text-to-Speech (TTS) and Speech-to-Text (STT) capabilities. The API integrates multiple TTS architectures including F5-TTS-MLX, XTTS, and Piper for speech synthesis, as well as lightning-whisper-mlx for speech recognition. This comprehensive solution provides high-quality speech processing with various customization options and enhancements.

## Features

- 🎯 Multiple TTS Architecture Support:
  - F5-TTS-MLX for high-quality neural TTS
  - XTTS for multilingual support
  - Piper for fast and efficient TTS
- 🎤 Speech-to-Text Support:
  - Lightning Whisper MLX for fast and accurate transcription
  - Multiple model sizes from tiny to large-v3
  - Optional quantization for improved performance
- 🔊 Voice Cloning Capabilities
- 🌐 Multi-language Support
- 🎨 Customizable Voice Settings
- 🎵 Audio Quality Enhancements
- 🚀 FastAPI-based REST API

## Prerequisites

- Python 3.9+
- FFmpeg installed on your system
- CUDA-capable GPU (optional, for improved performance)
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Goekdeniz-Guelmez/OpenAudioAPI.git
cd OpenAudioAPI
```

2: Install the required dependencies:

```bash
pip install -r requirements.txt
```

3: Install FFmpeg (if not already installed):

- On macOS: `brew install ffmpeg`
- On Ubuntu: `sudo apt-get install ffmpeg`
- On Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

## Configuration

The API uses a `config.json` file to manage both TTS and STT settings. The configuration is divided into two main sections: "TTS" for Text-to-Speech and "STT" for Speech-to-Text. Here's a detailed guide on configuring different architectures:

### Configuration Structure

The configuration file is structured as follows:

```json
{
  "TTS": {
    "tts-1-hd": {
      "voice_name": {
        // TTS voice configurations
      }
    },
    "tts-1": {
      // Additional TTS configurations
    }
  },
  "STT": {
    "model_name": {
      // STT model configurations
    }
  }
}
```

### Common Settings (TTS Architectures)

These settings can be applied to any voice configuration:

```json
{
  "enhance_quality": true/false,    // Enable audio enhancements
  "sample_rate": 44100,            // Audio sample rate
  "normalization_level": -3.0,     // Target dB for normalization
  "high_pass_filter": true/false,  // Remove low frequencies
  "noise_reduction": true/false    // Apply noise reduction
}
```

### F5-MLX Configuration

F5-MLX is optimized for high-quality speech synthesis with voice cloning capabilities.

```json
{
  "tts-1-hd": {
    "voice_name": {
      "architecture": "f5-mlx",
      "model": "path/to/model",
      "steps": 8,                    // Number of generation steps (higher = better quality)
      "cfg_strength": 2.0,           // Classifier-free guidance strength
      "sway_sampling_coef": -1.0,    // Sampling coefficient
      "speed": 1.0,                  // Speech speed (1.0 = normal)
      "method": "rk4",               // Generation method: "euler", "midpoint", or "rk4"
      "duration": null,              // Optional fixed duration in seconds
      "seed": null,                  // Random seed for reproducibility
      "quantization_bits": null,     // Audio quantization bits
      "ref_audio_path": "path/to/reference.wav",  // Required for voice cloning
      "ref_audio_text": "Reference text"          // Required for voice cloning
    }
  }
}
```

### XTTS Configuration

XTTS excels at multilingual speech synthesis with natural-sounding results.

```json
{
  "tts-1-hd": {
    "voice_name": {
      "architecture": "xtts",
      "model": "path/to/model",
      "language": "en",              // Language code or "auto" for detection
      "ref_audio_path": "path/to/reference.wav",  // Required
      "emotion": "neutral",          // Optional emotion parameter
      "speed": 1.0,                 // Speech speed
      "split_sentences": true       // Enable sentence splitting
    }
  }
}
```

### Piper Configuration

Piper provides fast and efficient TTS with good quality output.

```json
{
  "tts-1": {
    "voice_name": {
      "architecture": "piper",
      "model": "path/to/model",
      "length_scale": 1.0,          // Controls speech duration
      "noise_scale": 0.667,         // Affects voice variation
      "noise_w": 0.8,              // Affects voice consistency
      "volume": 1.0,               // Output volume
      "use_cuda": false,           // Enable GPU acceleration
      "speaker": null              // Optional speaker ID
    }
  }
}
```

### Whisper MLX Configuration

Lightning Whisper MLX provides fast and accurate speech recognition with various model sizes and quantization options.

Available Models:

- tiny
- small
- distil-small.en
- base
- medium
- distil-medium.en
- large
- large-v2
- distil-large-v2
- large-v3
- distil-large-v3

Quantization Options:

- None (default)
- "4bit"
- "8bit"

```json
{
  "STT": {
    "whisper-tiny": {
      "architecture": "whisper-mlx",
      "model": "tiny"
    },
    "whisper-large-v3": {
      "architecture": "whisper-mlx",
      "model": "large-v3",
      

### Generate Speech
```http
POST /v1/audio/speech
```

Request body parameters:

```json
{
  "model": "tts-1-hd",           // Model type
  "input": "Text to synthesize", // Input text
  "voice": "voice_name",         // Voice configuration to use
  "response_format": "wav",      // Output format: "wav" or "mp3"
  // ... additional parameters matching config options
}
```

### List Available Voices

```http
GET /v1/available_voices
```

Returns the complete configuration with available voices and their settings.

### Health Check

```http
GET /health
```

Returns the API health status.

## Audio Enhancement

When using the `tts-1-hd` model with `enhance_quality: true`, the following enhancements are applied:

- Sample rate conversion to specified rate (default: 44100Hz)
- High-pass filtering to remove low-frequency noise (optional)
- Noise reduction using FFT-based denoising (optional)
- Audio normalization to target level
- High-quality MP3 encoding for MP3 output format

## Examples

### Basic Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "tts-1-hd",
        "input": "Hello, world!",
        "voice": "default_voice",
        "response_format": "wav"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### Voice Cloning Example

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "tts-1-hd",
        "input": "This is a cloned voice speaking.",
        "voice": "cloned_voice",
        "response_format": "mp3",
        "enhance_quality": true
    }
)

with open("cloned_voice.mp3", "wb") as f:
    f.write(response.content)
```

### Whisper MLX

```python
import requests

files = {
  'file': open('/Users/gokdenizgulmez/Desktop/OpenAudioAPI/SJ.wav', 'rb')
}

data = {
  'model': 'whisper-tiny'
}
response = requests.post(url, files=files, data=data)

print(f"Response: {response.json()}")
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- 400: Bad Request (invalid parameters)
- 500: Internal Server Error (generation or processing failed)
- 200: Success

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## Acknowledgments

- [f5-tts-mlx by Lucas Newman](https://github.com/lucasnewman/f5-tts-mlx.git)
- [TTS by Coqui](https://github.com/coqui-ai/TTS.git)
- [TTS by Rhasspy](https://github.com/rhasspy/piper.git)
- [lightning-whisper-mlx by Mustafa Aljadery](https://github.com/mustafaaljadery/lightning-whisper-mlx.git)
- FastAPI
- FFmpeg

## Future PLans

- [ ] API endpoint to clone a voice via request.
- [ ] Supporting CUDA and Pytorch Whisper.
- [ ] Supporting Bark models.
- [ ] Supprting Piper TTS for 'tts-1'.
- [ ] Adding 'keep_alive' parameter to keep models loaded in RAM.

## Citing OpenAudioAPI

The OpenAudioAPI software suite was developed by Gökdeniz Gülmez. If you find
OpenAudioAPI useful in your research and wish to cite it, please use the following
BibTex entry:

```text
@software{
  OpenAudioAPI,
  author = {Gökdeniz Gülmez},
  title = {{OpenAudioAPI}: A versatile Speech API that supports both Text-to-Speech (TTS) and Speech-to-Text (STT) capabilities.},
  url = {https://github.com/Goekdeniz-Guelmez/OpenAudioAPI.git},
  version = {0.0.1},
  year = {2024},
}
```
