import os
import torch
import scipy.io.wavfile
import soundfile as sf

from diffusers import AudioLDM2Pipeline
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)
from datasets import load_dataset


class AudioGenerator:
    def __init__(self, model_id="cvssp/audioldm2", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"Loading models on {self.device}...")

        # ----------------------------
        # AudioLDM2 for music / sfx
        # ----------------------------
        from transformers import GPT2LMHeadModel

        lm = GPT2LMHeadModel.from_pretrained(
            model_id,
            subfolder="language_model",
            torch_dtype=self.dtype,
        )

        self.audioldm_pipe = AudioLDM2Pipeline.from_pretrained(
            model_id,
            language_model=lm,
            torch_dtype=self.dtype,
        )
        self.audioldm_pipe.to(self.device)

        if self.device == "cuda":
            self.audioldm_pipe.enable_model_cpu_offload()

        # ----------------------------
        # SpeechT5 for speech
        # ----------------------------
        print("Loading SpeechT5...")
        self.processor = SpeechT5Processor.from_pretrained(
            "microsoft/speecht5_tts"
        )
        self.speech_model = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts"
        ).to(self.device)

        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan"
        ).to(self.device)

        print("Loading speaker embeddings...")
        self.embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors",
            split="validation"
        )

        self.voice_map = {
            "male": 1200,
            "female": 7306
        }

    def _generate_speech(self, prompt, output_path, gender="female"):
        print("Gender",gender)
        inputs = self.processor(
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        speaker_id = self.voice_map.get(gender.lower(), 7306)

        speaker_embeddings = torch.tensor(
            self.embeddings_dataset[speaker_id]["xvector"]
        ).unsqueeze(0).to(self.device)

        speech = self.speech_model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings,
            vocoder=self.vocoder
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, speech.cpu().numpy(), 16000)

        return output_path

    def _generate_audio_audioldm(
        self,
        prompt,
        output_path,
        num_inference_steps=70,
        audio_length_in_s=6.0,
        domain="sfx"
    ):
        seed = torch.randint(0, 100000, (1,)).item()
        generator = torch.Generator(self.device).manual_seed(seed)

        guidance = 6.5 if domain == "sfx" else 6.0
        audio_length = 7.0 if domain == "sfx" else audio_length_in_s

        enhanced_prompt = (
            f"{prompt}. High quality, realistic, clear audio."
        )

        audio = self.audioldm_pipe(
            enhanced_prompt,
            num_inference_steps=num_inference_steps,
            audio_length_in_s=audio_length,
            guidance_scale=guidance,
            generator=generator,
        ).audios[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        scipy.io.wavfile.write(output_path, rate=16000, data=audio)

        return output_path

    def generate(
        self,
        prompt,
        output_path,
        domain,
        gender="female",
        num_inference_steps=70,
        audio_length_in_s=6.0,
    ):
        print(f"Generating [{domain}] → {prompt}")

        if domain == "speech":
            return self._generate_speech(
                prompt,
                output_path,
                gender
            )

        return self._generate_audio_audioldm(
            prompt,
            output_path,
            num_inference_steps,
            audio_length_in_s,
            domain
        )
