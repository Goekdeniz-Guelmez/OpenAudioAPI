# OpenAudioAPI

A versatile Text-to-Speech and Speech-To-Text API that supports multiple TTS architectures including F5-TTS-MLX, XTTS, and Piper. This API provides high-quality speech synthesis with various voice customization options and audio enhancements.

## Features

- 🎯 Multiple TTS Architecture Support:
  - F5-TTS-MLX for high-quality neural TTS
  - XTTS for multilingual support
  - Piper for fast and efficient TTS
- 🔊 Voice Cloning Capabilities
- 🌐 Multi-language Support
- 🎨 Customizable Voice Settings
- 🎵 Audio Quality Enhancements
- 🚀 FastAPI-based REST API

## Prerequisites

- Python 3.8+
- FFmpeg installed on your system
- CUDA-capable GPU (optional, for improved performance)
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg (if not already installed):
- On macOS: `brew install ffmpeg`
- On Ubuntu: `sudo apt-get install ffmpeg`
- On Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

## Configuration

The API uses a `config.json` file to manage voice settings and model configurations. Here's a detailed guide on configuring different TTS architectures:

### Common Settings (All Architectures)

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

## API Endpoints

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

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- 400: Bad Request (invalid parameters)
- 500: Internal Server Error (generation or processing failed)
- 200: Success

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

[Your chosen license]

## Acknowledgments

- F5-TTS-MLX
- XTTS
- Piper TTS
- FastAPI
- FFmpeg

## Support

For support, please [create an issue](your-issue-tracker-url) or contact [your-contact-info].