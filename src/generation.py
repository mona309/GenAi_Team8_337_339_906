"""
generation.py  (v2 — improved)
================================
Key changes vs v1
-----------------
1. SpeechT5 WAV saved as int16 PCM (was float32)
   soundfile.write() with a float32 array produces a FLOAT subtype WAV.
   Whisper's HuggingFace pipeline silently rejects FLOAT WAV files and
   returns an empty transcript — the root cause of semantic_score = 0.0
   for all speech samples in the original batch run.

   Fix: clip and cast the float32 array to int16 before writing.
   The output is identical audibly; only the on-disk format changes.

2. AudioLDM2 steps routing for complex ambient prompts
   Multi-element ambient scenes (coffee shop, city intersection) now use
   more diffusion steps by default. A simple keyword check in generate()
   raises the step count for prompts that contain known complex-scene words.

3. All other logic unchanged (emotion blending, speaker embeddings, etc.)
"""

import os
import re
import torch
import scipy.io.wavfile
import soundfile as sf
import numpy as np

from diffusers import AudioLDM2Pipeline
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Emotion configuration for SpeechT5  (unchanged from v1)
# ---------------------------------------------------------------------------

EMOTION_SPEAKER_MAP = {
    "female": {
        "neutral":   {"speaker_id": 7306, "blend": 0.00},
        "happy":     {"speaker_id": 7021, "blend": 0.35},
        "sad":       {"speaker_id": 6670, "blend": 0.40},
        "angry":     {"speaker_id": 7100, "blend": 0.45},
        "calm":      {"speaker_id": 6800, "blend": 0.30},
        "excited":   {"speaker_id": 7050, "blend": 0.40},
        "fearful":   {"speaker_id": 6750, "blend": 0.35},
    },
    "male": {
        "neutral":   {"speaker_id": 1200, "blend": 0.00},
        "happy":     {"speaker_id": 1150, "blend": 0.35},
        "sad":       {"speaker_id": 1300, "blend": 0.40},
        "angry":     {"speaker_id": 1250, "blend": 0.45},
        "calm":      {"speaker_id": 1100, "blend": 0.30},
        "excited":   {"speaker_id": 1175, "blend": 0.40},
        "fearful":   {"speaker_id": 1225, "blend": 0.35},
    },
}

EMOTION_TEXT_TRANSFORMS: dict[str, list[tuple]] = {
    "neutral": [],
    "happy": [
        (r"\.\s*$",  "!"),
        (r",\s*",    ", "),
    ],
    "sad": [
        (r"\.\s*$",  "..."),
        (r",\s*",    "... "),
        (r"!\s*",    "."),
    ],
    "angry": [
        (r"\b(\w+)\b", lambda m: m.group(1).upper() if len(m.group(1)) > 3 else m.group(1)),
        (r"\.\s*$",  "!"),
    ],
    "calm": [
        (r"!\s*",    "."),
        (r",\s*",    ", "),
    ],
    "excited": [
        (r"\.\s*$",  "!"),
        (r"(\w{5,})", lambda m: m.group(1)),
    ],
    "fearful": [
        (r"\.\s*$",  "..."),
        (r",\s*",    ", uh, "),
    ],
}

# Keywords that indicate a complex multi-layered ambient scene.
# AudioLDM2 benefits from more diffusion steps for these.
_COMPLEX_AMBIENT_KEYWORDS = {
    "coffee", "cafe", "restaurant", "intersection", "city", "crowd",
    "chatter", "subway", "office", "market", "station", "airport",
}

# Default and boosted step counts for AudioLDM2
_DEFAULT_AMBIENT_STEPS = 140
_COMPLEX_AMBIENT_STEPS = 200


def apply_emotion_transforms(text: str, emotion: str) -> str:
    transforms = EMOTION_TEXT_TRANSFORMS.get(emotion.lower(), [])
    result = text
    for pattern, replacement in transforms:
        try:
            result = re.sub(pattern, replacement, result)
        except Exception:
            pass
    return result


class AudioGenerator:
    def __init__(self, model_id="cvssp/audioldm2", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32

        print(f"Loading models on {self.device}...")

        # ── AudioLDM2 ────────────────────────────────────────────────────
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

        # ── SpeechT5 ─────────────────────────────────────────────────────
        print("Loading SpeechT5...")
        self.processor    = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.speech_model = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts"
        ).to(self.device)
        self.vocoder      = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan"
        ).to(self.device)

        print("Loading speaker embeddings...")
        self.embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors", split="validation"
        )

        self.voice_map = {"male": 1200, "female": 7306}

    # ── Speaker embedding helpers ─────────────────────────────────────────────

    def _get_speaker_embedding(
        self,
        gender: str,
        emotion: str = "neutral",
    ) -> torch.Tensor:
        gender_lower  = gender.lower()
        emotion_lower = emotion.lower()

        emotion_cfg = (
            EMOTION_SPEAKER_MAP
            .get(gender_lower, EMOTION_SPEAKER_MAP["female"])
            .get(emotion_lower, EMOTION_SPEAKER_MAP[gender_lower]["neutral"])
        )

        base_id  = self.voice_map.get(gender_lower, 7306)
        blend_id = emotion_cfg["speaker_id"]
        alpha    = emotion_cfg["blend"]

        base_vec = np.array(self.embeddings_dataset[base_id]["xvector"], dtype=np.float32)

        if alpha > 0.0 and blend_id != base_id:
            blend_vec = np.array(
                self.embeddings_dataset[blend_id]["xvector"], dtype=np.float32
            )
            combined = (1.0 - alpha) * base_vec + alpha * blend_vec
        else:
            combined = base_vec

        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        return torch.tensor(combined).unsqueeze(0).to(self.device)

    # ── Speech generation ─────────────────────────────────────────────────────

    def _generate_speech(
        self,
        prompt: str,
        output_path: str,
        gender: str = "female",
        emotion: str = "neutral",
    ) -> str:
        """
        Generate speech with optional emotional colouring.

        FIX (v2): Save WAV as int16 PCM instead of float32.
        Whisper's HuggingFace pipeline silently returns an empty transcript
        when given a FLOAT WAV file. Casting to int16 before writing fixes
        this and enables proper WER-based semantic scoring.
        """
        emotion_lower = emotion.lower() if emotion else "neutral"

        # Step 1 — prosody text preprocessing
        processed_text = apply_emotion_transforms(prompt, emotion_lower)
        if processed_text != prompt:
            print(f"[Speech] Emotion '{emotion_lower}' text transform: "
                  f"{prompt!r} → {processed_text!r}")

        # Step 2 — speaker embedding with emotion blend
        speaker_embeddings = self._get_speaker_embedding(gender, emotion_lower)
        print(f"[Speech] gender={gender} emotion={emotion_lower}")

        # Step 3 — inference
        #
        # SpeechT5 truncation fix
        # -----------------------
        # SpeechT5Processor uses a SentencePiece tokenizer whose
        # model_max_length is 512 tokens. Calling processor(text=...) with
        # no explicit argument silently truncates anything beyond token 512,
        # dropping the tail of the text without any warning.
        #
        # For typical TTS prompts (one or two sentences) this never triggers.
        # But RAG-enhanced prompts that accidentally include extra descriptors
        # can exceed 512 tokens. We guard against this in two ways:
        #
        #   1. Warn loudly when the raw character count suggests the prompt
        #      is approaching the limit (>400 chars ≈ ~100 tokens is a rough
        #      heuristic; SpeechT5 averages ~4 chars/token for English).
        #
        #   2. Pass truncation=True and max_length=500 explicitly so
        #      truncation is deliberate and auditable rather than silent.
        #      We use 500 (not 512) to leave room for any special tokens
        #      the processor prepends/appends.
        #
        # Note: for speech, rag_enhancer already passes prompts through
        # unchanged, so this guard is a safety net for future changes.

        _SPEECHT5_MAX_TOKENS = 500   # conservative — model limit is 512
        _CHARS_PER_TOKEN     = 4     # rough English average for this tokenizer

        if len(processed_text) > _SPEECHT5_MAX_TOKENS * _CHARS_PER_TOKEN:
            # Tokenise first to get the real count, then warn if needed
            probe = self.processor(
                text=processed_text,
                return_tensors="pt",
                truncation=False,
            )
            n_tokens = probe["input_ids"].shape[-1]
            if n_tokens > _SPEECHT5_MAX_TOKENS:
                import warnings as _w
                _w.warn(
                    f"[SpeechT5] Prompt is {n_tokens} tokens — exceeds the "
                    f"{_SPEECHT5_MAX_TOKENS}-token safe limit. "
                    f"Text will be truncated; tail content will NOT be spoken. "
                    f"Shorten the prompt or split it into multiple segments.\n"
                    f"  Prompt: {processed_text[:120]!r}…"
                )

        inputs = self.processor(
            text=processed_text,
            return_tensors="pt",
            truncation=True,
            max_length=_SPEECHT5_MAX_TOKENS,
        ).to(self.device)

        speech = self.speech_model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings,
            vocoder=self.vocoder,
        )

        # Step 4 — save as int16 PCM (CRITICAL FIX)
        # sf.write with a float32 array produces a FLOAT subtype WAV that
        # Whisper rejects. Convert to int16 first.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        speech_np    = speech.cpu().numpy()
        speech_int16 = (speech_np * 32767.0).clip(-32768, 32767).astype(np.int16)
        sf.write(output_path, speech_int16, samplerate=16000, subtype="PCM_16")

        print(f"[Speech] Saved int16 PCM WAV → {output_path}")
        return output_path

    # ── AudioLDM2 generation ──────────────────────────────────────────────────

    def _get_ambient_steps(self, prompt: str, requested_steps: int) -> int:
        """
        Boost diffusion steps for complex multi-element ambient prompts.
        Complex scenes need more steps to balance competing audio elements.
        """
        prompt_words = set(prompt.lower().split())
        if prompt_words & _COMPLEX_AMBIENT_KEYWORDS:
            boosted = max(requested_steps, _COMPLEX_AMBIENT_STEPS)
            if boosted > requested_steps:
                print(f"[Generate] Complex ambient detected — steps {requested_steps} → {boosted}")
            return boosted
        return requested_steps

    def _generate_audio_audioldm(
        self,
        prompt: str,
        output_path: str,
        num_inference_steps: int = 70,
        audio_length_in_s: float = 6.0,
        domain: str = "sfx",
    ) -> str:
        seed      = torch.randint(0, 100000, (1,)).item()
        generator = torch.Generator(self.device).manual_seed(seed)

        guidance     = 6.5 if domain == "sfx" else 6.0
        audio_length = 7.0 if domain == "sfx" else audio_length_in_s

        # Boost steps for complex ambient prompts
        if domain == "ambient":
            num_inference_steps = self._get_ambient_steps(prompt, num_inference_steps)

        enhanced_prompt = f"{prompt}. High quality, realistic, clear audio."

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

    # ── Public generate() ─────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        output_path: str,
        domain: str,
        gender: str = "female",
        emotion: str = "neutral",
        num_inference_steps: int = 70,
        audio_length_in_s: float = 6.0,
    ) -> str:
        """
        Route to the correct model based on domain.

        Parameters
        ----------
        prompt               : Text description / TTS text.
        output_path          : Destination WAV path.
        domain               : "speech" | "music" | "sfx" | "ambient"
        gender               : "female" | "male"  (speech only)
        emotion              : "neutral" | "happy" | "sad" | "angry" |
                               "calm" | "excited" | "fearful"  (speech only)
        num_inference_steps  : AudioLDM2 diffusion steps (ignored for speech).
        audio_length_in_s    : Clip length in seconds (ignored for speech).
        """
        print(f"[Generate] domain={domain} | {prompt[:80]!r}")

        if domain == "speech":
            return self._generate_speech(prompt, output_path, gender, emotion)

        return self._generate_audio_audioldm(
            prompt, output_path, num_inference_steps, audio_length_in_s, domain
        )
