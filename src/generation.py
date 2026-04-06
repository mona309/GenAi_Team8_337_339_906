import os
import torch
import scipy.io.wavfile
from diffusers import AudioLDM2Pipeline

class AudioGenerator:
    def __init__(self, model_id="cvssp/audioldm2", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading AudioLDM2 on {self.device}...")

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        from transformers import GPT2LMHeadModel
        lm = GPT2LMHeadModel.from_pretrained(
            model_id,
            subfolder="language_model",
            torch_dtype=dtype
        )

        self.pipe = AudioLDM2Pipeline.from_pretrained(
            model_id,
            language_model=lm,
            torch_dtype=dtype
        )

        self.pipe.to(self.device)

        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()

    def generate(self, prompt: str, output_path: str,
                 num_inference_steps: int = 70,
                 audio_length_in_s: float = 6.0):  # ✅ FIXED

        print(f"Generating audio for prompt: '{prompt}'")

        seed = torch.randint(0, 100000, (1,)).item()
        generator = torch.Generator(self.device).manual_seed(seed)

        # 🔥 Domain-specific tuning
        if "sfx" in output_path:
            audio_length = 7.0
            guidance = 6.5
        else:
            audio_length = audio_length_in_s
            guidance = 6.0

        enhanced_prompt = f"{prompt}. High quality, realistic, clear audio, professional recording."

        audio = self.pipe(
            enhanced_prompt,
            num_inference_steps=num_inference_steps,
            audio_length_in_s=audio_length,
            guidance_scale=guidance,
            generator=generator
        ).audios[0]

        sample_rate = 16000

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        scipy.io.wavfile.write(output_path, rate=sample_rate, data=audio)

        print(f"Saved: {output_path}")
        return output_path


if __name__ == "__main__":
    generator = AudioGenerator()
    generator.generate(
        "A realistic thunderstorm with rain and wind, immersive environmental sound",
        "outputs/audio/test.wav"
    )
