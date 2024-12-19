from huggingface_hub import hf_hub_download
from lightning_whisper_mlx import transcribe
from typing import Optional
import os

WHISPER_MODELS = {
    "tiny": {
        "base": "mlx-community/whisper-tiny", 
        "4bit": "mlx-community/whisper-tiny-mlx-4bit", 
        "8bit": "mlx-community/whisper-tiny-mlx-8bit"
    }, 
    "small": {
        "base": "mlx-community/whisper-small-mlx", 
        "4bit": "mlx-community/whisper-small-mlx-4bit", 
        "8bit": "mlx-community/whisper-small-mlx-8bit"
    },
    "distil-small.en": {
        "base": "mustafaaljadery/distil-whisper-mlx", 
    },
    "base": {
        "base" : "mlx-community/whisper-base-mlx", 
        "4bit" : "mlx-community/whisper-base-mlx-4bit",
        "8bit" : "mlx-community/whisper-base-mlx-8bit"
    },
    "medium": {
        "base": "mlx-community/whisper-medium-mlx",
        "4bit": "mlx-community/whisper-medium-mlx-4bit", 
        "8bit": "mlx-community/whisper-medium-mlx-8bit"
    }, 
    "distil-medium.en": {
        "base": "mustafaaljadery/distil-whisper-mlx", 
    }, 
    "large": {
        "base": "mlx-community/whisper-large-mlx", 
        "4bit": "mlx-community/whisper-large-mlx-4bit", 
        "8bit": "mlx-community/whisper-large-mlx-8bit", 
    },
    "large-v2": {
        "base": "mlx-community/whisper-large-v2-mlx",
        "4bit": "mlx-community/whisper-large-v2-mlx-4bit",
        "8bit": "mlx-community/whisper-large-v2-mlx-8bit", 
    },
    "distil-large-v2": {
        "base": "mustafaaljadery/distil-whisper-mlx",
    },
    "large-v3": {
        "base": "mlx-community/whisper-large-v3-mlx",
        "4bit": "mlx-community/whisper-large-v3-mlx-4bit",
        "8bit": "mlx-community/whisper-large-v3-mlx-8bit", 
    },
    "distil-large-v3": {
        "base": "mustafaaljadery/distil-whisper-mlx",
    },
}

class CustomLightningWhisperMLX:
    def __init__(self, model_name: str, batch_size: int = 4, quant: Optional[str] = None):
        if quant and (quant != "4bit" and quant != "8bit"):
            raise ValueError("Quantization must be `4bit` or `8bit`")
        
        self.name = model_name
        self.batch_size = batch_size
        
        # First try: Check if it's in our predefined models
        if model_name in WHISPER_MODELS:
            if quant and "distil" not in model_name:
                repo_id = WHISPER_MODELS[model_name][quant]
            else:
                repo_id = WHISPER_MODELS[model_name]['base']
                
        else:
            # Second try: Check if it's a local path
            if os.path.exists(model_name):
                self.local_model = True
                return  # If it's a local path, we're done
                
            # Third try: Treat it as a HuggingFace repo ID
            try:
                repo_id = model_name  # Try using the model_name directly as a repo ID
                # Test if repo exists by attempting to download config
                hf_hub_download(repo_id=repo_id, filename="config.json", local_dir="./tmp_test")
                os.remove("./tmp_test/config.json")  # Clean up test file
                os.rmdir("./tmp_test")
            except Exception as e:
                raise ValueError(
                    f"Model '{model_name}' is not in predefined models, "
                    "is not a valid local path, and "
                    "is not a valid HuggingFace repository."
                )
        
        # If we're using a local model, skip the download process
        if hasattr(self, 'local_model') and self.local_model:
            return

        # Handle distil model naming for predefined models
        if model_name in WHISPER_MODELS and quant and "distil" in model_name:
            if quant == "4bit":
                self.name += "-4-bit"
            else:
                self.name += "-8-bit"

        # Set up file paths based on model type
        if model_name in WHISPER_MODELS and "distil" in model_name:
            filename1 = f"./voices/models/mlx/{self.name}/weights.npz"
            filename2 = f"./voices/models/mlx/{self.name}/config.json"
            local_dir = "./"
        else:
            filename1 = "weights.npz"
            filename2 = "config.json"
            local_dir = f"./voices/models/mlx/{self.name}"

        # Download model files
        os.makedirs(local_dir, exist_ok=True)
        hf_hub_download(repo_id=repo_id, filename=filename1, local_dir=local_dir)
        hf_hub_download(repo_id=repo_id, filename=filename2, local_dir=local_dir)

    def transcribe(self, audio_path: str, **kwargs) -> dict:
        """
        Transcribe audio using all available parameters
        """
        return transcribe.transcribe_audio(
            audio=audio_path,
            path_or_hf_repo=f'./voices/models/mlx/{self.name}',
            batch_size=self.batch_size,
            **kwargs
        )