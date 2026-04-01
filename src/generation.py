import os
import torch
import scipy.io.wavfile
from diffusers import AudioLDM2Pipeline

class AudioGenerator:
    def __init__(self, model_id="cvssp/audioldm2", device="cuda"):
        """
        Initializes the Audio Generation Pipeline.
        AudioLDM 2 uses Flan-T5 for conditioning naturally, fulfilling the Flan-T5 GenAI objective.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading AudioLDM2 on {self.device}...")
        
        # Load pipeline in fp16 if on GPU to save VRAM
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Override the language model to fix compatibility bug in newer diffusers
        # Older models specified 'GPT2Model' but diffusers now requires 'GPT2LMHeadModel'
        from transformers import GPT2LMHeadModel
        print("Initializing correct GPT2LMHeadModel to resolve pipeline expected types...")
        lm = GPT2LMHeadModel.from_pretrained(model_id, subfolder="language_model", torch_dtype=dtype)
        
        self.pipe = AudioLDM2Pipeline.from_pretrained(model_id, language_model=lm, torch_dtype=dtype)
        
        self.pipe.to(self.device)
        
        # Optional: enable offload for memory optimization
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()

    def generate(self, prompt: str, output_path: str, num_inference_steps: int = 25, audio_length_in_s: float = 3.0):
        """
        Generates audio from a given prompt and saves it as a WAV file.
        Returns the path to the generated file.
        """
        print(f"Generating audio for prompt: '{prompt}'")
        
        # Generate raw audio numpy arrays
        generator = torch.Generator(self.device).manual_seed(42)
        audio = self.pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            audio_length_in_s=audio_length_in_s,
            generator=generator
        ).audios[0] # Take first item from batch
        
        # Extract audio sample rate natively from pipeline configs
        sample_rate = 16000 # AudioLDM2 outputs @ 16kHz
        
        # Save audio
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        scipy.io.wavfile.write(output_path, rate=sample_rate, data=audio)
        print(f"Saved: {output_path}")
        
        return output_path

if __name__ == "__main__":
    generator = AudioGenerator()
    generator.generate("A loud thunderclap followed by heavy rain hitting a roof.", "test_thunder.wav")
