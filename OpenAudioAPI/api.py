from fastapi import FastAPI, HTTPException, UploadFile
from fastapi import File
from fastapi.responses import Response
from pydantic import BaseModel
import tempfile
import os
import uuid
import json
import torch
import subprocess
from TTS.api import TTS
from f5_tts_mlx.generate import generate, SAMPLE_RATE
from langdetect import detect
from typing import Optional, Literal, Union, List, Tuple
from handlers import CustomLightningWhisperMLX

app = FastAPI(title="OpenAudioAPI")

# Initialize models dict to store XTTS models
device = "cuda" if torch.cuda.is_available() else "cpu"
xtts_models = {}
whisper_models = {}

def get_xtts_model(model_path: str):
    """Get or initialize XTTS model for given path"""
    if model_path not in xtts_models:
        xtts_models[model_path] = TTS(model_path).to(device)
    return xtts_models[model_path]

# Load config file
def load_config():
    with open('voices/config.json', 'r') as f:
        return json.load(f)

def get_voice_config(voice_name: str, model_type: str = "tts-1-hd"):
    """Get voice configuration from the updated config structure"""
    config = load_config()
    
    # Check if TTS section exists
    if "TTS" not in config:
        raise HTTPException(400, "TTS configuration not found in config")
    
    # Check if model type exists under TTS
    if model_type not in config["TTS"]:
        raise HTTPException(400, f"Model type {model_type} not found in TTS config")
    
    # Check if voice exists under model type
    if voice_name not in config["TTS"][model_type]:
        raise HTTPException(400, f"Voice {voice_name} not found in config for model {model_type}")
    
    return config["TTS"][model_type][voice_name]

def get_stt_config(model_name: str):
    """Get STT model configuration"""
    config = load_config()
    if "STT" not in config:
        raise HTTPException(400, "STT configuration not found in config")
    
    if model_name not in config["STT"]:
        raise HTTPException(400, f"Model {model_name} not found in STT config")
    
    return config["STT"][model_name]

def check_and_convert_ref_audio(ref_audio_path: str, temp_dir: str, voice_name: str, model_type: str) -> str:
    """
    Check reference audio sample rate and convert to 24kHz if necessary.
    Updates config.json if conversion is needed.
    """
    if not ref_audio_path:
        raise HTTPException(400, "Reference audio path is required for f5-mlx model")
        
    if not os.path.exists(ref_audio_path):
        raise HTTPException(400, f"Reference audio file not found: {ref_audio_path}")

    try:
        # Get audio file info using ffprobe
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=sample_rate',
            '-of', 'json',
            ref_audio_path
        ]
        
        probe_output = subprocess.run(
            probe_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        info = json.loads(probe_output.stdout)
        if not info.get('streams'):
            raise HTTPException(400, f"No audio stream found in file: {ref_audio_path}")
            
        current_sample_rate = int(info['streams'][0]['sample_rate'])
        
        # If already 24kHz, return original path
        if current_sample_rate == 24000:
            return ref_audio_path
            
        # Get the directory and base filename
        ref_audio_dir = os.path.dirname(ref_audio_path)
        if not ref_audio_dir:  # If ref_audio_path is just a filename
            ref_audio_dir = "."
            
        original_filename = os.path.basename(ref_audio_path)
        filename_without_ext = os.path.splitext(original_filename)[0]
        converted_filename = f"{filename_without_ext}_24k.wav"
        
        # Create full path for converted file in the same directory as original
        converted_path = os.path.join(ref_audio_dir, converted_filename)
        
        # Convert to 24kHz if converted file doesn't exist
        if not os.path.exists(converted_path):
            convert_cmd = [
                'ffmpeg', '-y',
                '-i', ref_audio_path,
                '-ar', '24000',
                '-ac', '1',
                '-acodec', 'pcm_s16le',
                converted_path
            ]
            
            result = subprocess.run(
                convert_cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise HTTPException(500, f"Failed to convert audio: {result.stderr}")
                
            if not os.path.exists(converted_path):
                raise HTTPException(500, "Failed to create converted audio file")
        
        # Update config.json with new path
        config_path = '/Users/gokdenizgulmez/Desktop/Florence5-api/voices/config.json'
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if model_type in config and voice_name in config[model_type]:
                config[model_type][voice_name]["ref_audio_path"] = converted_path
                if "original_ref_audio_path" not in config[model_type][voice_name]:
                    config[model_type][voice_name]["original_ref_audio_path"] = ref_audio_path
            
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
        except Exception as e:
            # Log config update error but don't fail the conversion
            print(f"Warning: Could not update config file: {str(e)}")
        
        return converted_path
        
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"Error checking audio sample rate: {str(e)}")
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Error parsing ffprobe output: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Error processing reference audio: {str(e)}")

class TTSRequest(BaseModel):
    # Base parameters
    # Audio quality settings
    sample_rate: int = 44100  # Higher sample rate for better quality
    enhance_quality: bool = False  # Enable audio enhancement
    normalization_level: float = -3.0  # Target dB for normalization
    high_pass_filter: bool = False  # Remove low frequency noise
    noise_reduction: bool = False  # Apply noise reduction
    model: str = "tts-1-hd"
    input: str
    voice: str | None = None
    response_format: str = "wav"
    
    # F5-TTS parameters
    speed: float = 1.0
    duration: float | None = None
    steps: int = 8
    method: str = "rk4"
    cfg_strength: float = 2.0
    sway_sampling_coef: float = -1.0
    seed: int | None = None
    quantization_bits: int | None = None
    ref_audio_path: str | None = None
    ref_audio_text: str | None = None
    
    # Piper-specific parameters
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8
    volume: float = 1.0
    use_cuda: bool = False
    speaker: int | None = None
    
    # XTTS-specific parameters
    language: str = "auto"
    emotion: str | None = None
    split_sentences: bool = True
    
    # Additional config parameters that might be in config.json
    temperature: float | None = None  # For models that support temperature
    top_p: float | None = None       # For models that support top_p sampling
    top_k: int | None = None         # For models that support top_k sampling
    repeat_penalty: float | None = None  # For preventing repetition
    presence_penalty: float | None = None  # For adjusting presence penalty
    frequency_penalty: float | None = None  # For adjusting frequency penalty

def generate_xtts(text: str, voice_config: dict, output_path: str):
    """Generate audio using XTTS model"""
    model_path = voice_config["model"]
    tts = get_xtts_model(model_path)
    language = voice_config.get("language", "en")
    if language == "auto":
        language = detect(text)
    speaker_wav = voice_config["ref_audio_path"]
    emotion = voice_config.get("emotion", None)
    speed = voice_config.get("speed", 1.0)
    split_sentences = voice_config.get("split_sentences", True)
    
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
        file_path=output_path,
        emotion=emotion,
        speed=speed,
        split_sentences=split_sentences
    )

def generate_f5(text: str, voice_config: dict, request: TTSRequest, output_path: str):
    """Generate audio using f5-tts-mlx"""
    # First, check if we're doing voice cloning or using the base model
    is_voice_cloning = "ref_audio_path" in voice_config or "ref_audio_text" in voice_config
    
    # Required parameters
    generate_kwargs = {
        "generation_text": text,
        "model_name": voice_config["model"],
        "output_path": output_path,
        "steps": voice_config.get("steps", 8),
        "cfg_strength": voice_config.get("cfg_strength", 2.0),
        "sway_sampling_coef": voice_config.get("sway_sampling_coef", -1.0),
        "speed": voice_config.get("speed", 1.0),
        "method": voice_config.get("method", "rk4")
    }
    
    if is_voice_cloning:
        # For voice cloning, we need both ref_audio_path and ref_audio_text
        if not voice_config.get("ref_audio_path"):
            raise ValueError("ref_audio_path is required for voice cloning")
        if not voice_config.get("ref_audio_text"):
            raise ValueError("ref_audio_text is required for voice cloning")
            
        generate_kwargs["ref_audio_path"] = voice_config["ref_audio_path"]
        generate_kwargs["ref_audio_text"] = voice_config["ref_audio_text"]
    
    # Optional parameters
    optional_params = {
        "duration": voice_config.get("duration", request.duration),
        "seed": voice_config.get("seed", request.seed),
        "quantization_bits": voice_config.get("quantization_bits", request.quantization_bits)
    }
    
    # Only add optional parameters if they are not None
    for key, value in optional_params.items():
        if value is not None:
            generate_kwargs[key] = value
    
    # Validate method
    if generate_kwargs["method"] not in ["euler", "midpoint", "rk4"]:
        raise ValueError(f"Invalid method {generate_kwargs['method']}. Must be one of: euler, midpoint, rk4")
    
    try:
        # Print kwargs for debugging
        print(f"Generating with kwargs: {generate_kwargs}")
        generate(**generate_kwargs)
    except Exception as e:
        print(f"Error details: {str(e)}")
        print(f"Generate kwargs: {generate_kwargs}")
        raise Exception(f"Error generating speech with f5-tts-mlx: {str(e)}")

def generate_piper(text: str, voice_config: dict, request: TTSRequest, output_path: str):
    """Generate audio using Piper TTS"""
    cmd = [
        "piper-tts",
        "--model", voice_config["model"],
        "--output_file", output_path,
        "--length_scale", str(voice_config.get("length_scale", request.length_scale)),
        "--noise_scale", str(voice_config.get("noise_scale", request.noise_scale)),
        "--noise_w", str(voice_config.get("noise_w", request.noise_w)),
        "--volume", str(voice_config.get("volume", request.volume))
    ]
    
    # Add speaker if specified in config
    if "speaker" in voice_config:
        cmd.extend(["--speaker", str(voice_config["speaker"])])
    
    # Add CUDA flag if specified
    if voice_config.get("use_cuda", request.use_cuda):
        cmd.append("--cuda")

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=text)
        
        if process.returncode != 0:
            raise Exception(f"Piper TTS error: {stderr}")
            
    except Exception as e:
        raise Exception(f"Error generating speech with Piper: {str(e)}")

class STT(BaseModel):
    model: str  # This will be the config key (e.g., "whisper-tiny")
    response_format: Literal["json", "text", "srt", "vtt", "verbose_json"] = "json"
    
    # Transcription parameters
    verbose: Optional[bool] = None
    temperature: Union[float, Tuple[float, ...]] = (0.0, 1.0)
    compression_ratio_threshold: Optional[float] = 2.4
    logprob_threshold: Optional[float] = -1
    no_speech_threshold: Optional[float] = 0.6
    condition_on_previous_text: bool = True
    initial_prompt: Optional[str] = None
    word_timestamps: bool = False
    prepend_punctuations: str = """\"'"¿([{-"""
    append_punctuations: str = """\"'.。,，!！?？:：")]}、"""
    clip_timestamps: Union[str, List[float]] = "0"
    hallucination_silence_threshold: Optional[float] = None
    language: Optional[str] = None
    batch_size: int = 2


def get_whisper_model(model_name: str, batch_size: int = 12, quant: Optional[str] = None) -> CustomLightningWhisperMLX:
    """Get or initialize Whisper model with specific configuration"""
    model_key = f"{model_name}_{batch_size}_{quant}"
    if model_key not in whisper_models:
        whisper_models[model_key] = CustomLightningWhisperMLX(
            model_name=model_name,
            batch_size=batch_size,
            quant=quant
        )
    return whisper_models[model_key]

@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """Create speech from text using various TTS models"""
    if request.response_format not in ["mp3", "wav"]:
        raise HTTPException(400, "Only mp3 and wav formats are supported")

    if not request.voice:
        raise HTTPException(400, "Voice name must be specified")

    voice_config = get_voice_config(request.voice, request.model)

    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = os.path.join(temp_dir, "output.wav")
        try:
            # Choose generation method based on model type
            archtitecure = voice_config.get("architecture", "f5-mlx")

            if not voice_config.get("architecture"):
                raise HTTPException(500, "Model Architecture is not defined in config file")

            if archtitecure == "xtts":
                generate_xtts(request.input, voice_config, wav_path)
            elif archtitecure == "piper":
                raise HTTPException(500, "'piper' is not implemented yet")
                # generate_piper(request.input, voice_config, request, wav_path)
            elif archtitecure == "f5-mlx":
                if "ref_audio_path" in voice_config:
                    voice_config["ref_audio_path"] = check_and_convert_ref_audio(
                        voice_config["ref_audio_path"], 
                        temp_dir,
                        request.voice,
                        request.model
                    )
                generate_f5(request.input, voice_config, request, wav_path)
            elif archtitecure == "bark":
                raise HTTPException(500, "'bark' is not implemented yet")
            else:
                raise HTTPException(500, "Model Architecture is not supported, only ['xtts', 'piper', 'f5-mlx'] are supported")

            if not os.path.exists(wav_path):
                raise HTTPException(500, "Failed to generate audio file")

            # Get audio enhancement settings from config or request
            enhance_quality = voice_config.get("enhance_quality", request.enhance_quality)
            sample_rate = voice_config.get("sample_rate", request.sample_rate)
            normalization_level = voice_config.get("normalization_level", request.normalization_level)
            high_pass_filter = voice_config.get("high_pass_filter", request.high_pass_filter)
            noise_reduction = voice_config.get("noise_reduction", request.noise_reduction)
            
            # Apply audio enhancements if requested and model is tts-1-hd
            if enhance_quality and request.model == "tts-1-hd":
                enhanced_path = os.path.join(temp_dir, "enhanced.wav")
                
                # Build ffmpeg command with quality enhancements
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-i', wav_path,
                    '-ar', str(sample_rate),  # Set sample rate
                    '-acodec', 'pcm_s16le'  # Use 16-bit PCM encoding
                ]
                
                # Add audio filters based on requested enhancements
                filters = []
                
                if high_pass_filter:
                    filters.append("highpass=f=100")  # Remove frequencies below 100Hz
                
                if noise_reduction:
                    filters.append("afftdn=nf=-25")  # Noise reduction
                
                # Normalization (always apply if enhance_quality is True)
                filters.append(f"loudnorm=I={normalization_level}:TP=-1.5:LRA=11")
                
                if filters:
                    ffmpeg_cmd.extend(['-af', ','.join(filters)])
                
                ffmpeg_cmd.append(enhanced_path)
                subprocess.run(ffmpeg_cmd, check=True)
                wav_path = enhanced_path  # Use enhanced version for further processing

            # Convert to MP3 if requested
            if request.response_format == "mp3":
                mp3_path = os.path.join(temp_dir, "output.mp3")
                
                mp3_cmd = [
                    'ffmpeg', '-y',
                    '-i', wav_path,
                    '-codec:a', 'libmp3lame',
                    '-q:a', '0',  # Highest quality VBR encoding
                    '-joint_stereo', '1'  # Better stereo encoding
                ]
                
                if request.model == "tts-1-hd":
                    # Add additional MP3 encoding parameters for HD quality
                    mp3_cmd.extend([
                        '-b:a', '320k',  # Maximum bitrate
                        '-compression_level', '0'  # Less compression
                    ])
                
                mp3_cmd.append(mp3_path)
                subprocess.run(mp3_cmd, check=True)
                output_path = mp3_path
            else:
                output_path = wav_path

            with open(output_path, 'rb') as f:
                audio_data = f.read()

            return Response(
                content=audio_data,
                media_type=f"audio/{request.response_format}",
                headers={
                    'Content-Disposition': f'attachment; filename=speech_{uuid.uuid4()}.{request.response_format}'
                }
            )
        except Exception as e:
            raise HTTPException(500, f"Error generating speech: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/v1/available_voices")
async def list_voices():
    config = load_config()
    return config

@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    request: STT = STT(model="whisper-base")
):
    """
    Create transcription from audio file using Whisper MLX
    """
    # Get model configuration from config.json
    model_config = get_stt_config(request.model)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file
        audio_path = os.path.join(temp_dir, file.filename)
        with open(audio_path, "wb") as f:
            f.write(await file.read())

        try:
            if model_config["architecture"] != "whisper-mlx":
                raise HTTPException(400, "Only whisper-mlx architecture is supported")

            # Initialize model with config parameters
            model_name = model_config["model"]
            batch_size = model_config.get("batch_size", request.batch_size)
            quant = model_config.get("quantization_bits", None)

            # Get or initialize the model
            whisper = get_whisper_model(model_name, batch_size, quant)

            # Prepare transcription parameters
            param_mapping = {
                "verbose": ("verbose", request.verbose),
                "temperature": ("temperature", request.temperature),
                "compression_ratio_threshold": ("compression_ratio_threshold", request.compression_ratio_threshold),
                "logprob_threshold": ("logprob_threshold", request.logprob_threshold),
                "no_speech_threshold": ("no_speech_threshold", request.no_speech_threshold),
                "condition_on_previous_text": ("condition_on_previous_text", request.condition_on_previous_text),
                "initial_prompt": ("initial_prompt", request.initial_prompt),
                "word_timestamps": ("word_timestamps", request.word_timestamps),
                "prepend_punctuations": ("prepend_punctuations", request.prepend_punctuations),
                "append_punctuations": ("append_punctuations", request.append_punctuations),
                "clip_timestamps": ("clip_timestamps", request.clip_timestamps),
                "hallucination_silence_threshold": ("hallucination_silence_threshold", request.hallucination_silence_threshold),
                "language": ("language", request.language),
            }

            transcribe_params = {
                param_name: model_config.get(config_key)
                for param_name, (config_key, _) in param_mapping.items()
                if model_config.get(config_key) is not None
            }

            # Override with request values if they are provided
            for param_name, (_, request_value) in param_mapping.items():
                if request_value is not None:
                    transcribe_params[param_name] = request_value

            # Perform transcription
            result = whisper.transcribe(audio_path, **transcribe_params)

            # Format response based on requested format
            if request.response_format == "text":
                return result["text"]
            elif request.response_format in ["srt", "vtt"]:
                return Response(
                    content=result["segments"],
                    media_type="text/plain",
                    headers={
                        "Content-Disposition": f"attachment; filename=transcription.{request.response_format}"
                    }
                )
            elif request.response_format == "verbose_json":
                return result
            else:  # default json format
                return {"text": result["text"]}

        except Exception as e:
            raise HTTPException(500, f"Error transcribing audio: {str(e)}")