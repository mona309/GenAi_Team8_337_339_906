"""
evaluation.py  (v2 — improved)
================================
Key changes vs v1
-----------------
1. Whisper diagnostic logging
   All speech samples had semantic_score = 0.0 in the original run because
   Whisper was producing empty transcripts silently. We now log the raw
   transcript and raise a clear warning when it is empty, so the operator
   can see exactly what went wrong.

2. WAV format validation before transcription
   SpeechT5 / sf.write can produce float32 WAV files. Whisper's HuggingFace
   pipeline rejects float32 audio and returns an empty transcript without
   raising an exception. We now check the file's dtype and log a warning
   when float32 is detected. The fix lives in generation.py (save as int16),
   but this guard surfaces the issue clearly if the file has the wrong format.

3. Graceful empty-transcript handling
   If _transcribe() returns an empty string we now log a warning AND set
   semantic_score to 0 explicitly (unchanged behaviour, but now visible).

4. evaluate_poas() passes original_prompt to CLAP, enhanced prompt to ASR
   When RAG is enabled, the text sent to CLAP is the full enhanced prompt
   (as before). But ASR compares the Whisper transcript against the
   *original* TTS text (what was actually spoken), not the long RAG string.
   A new optional parameter `asr_reference` lets the caller pass the
   original prompt separately. If omitted, text_prompt is used for both
   (backward-compatible).

5. No mock scores, no silent fallback (unchanged from v1 — kept)
"""

import os
import collections
import warnings
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import torch

try:
    import laion_clap
    HAS_CLAP = True
except ImportError:
    HAS_CLAP = False
    warnings.warn("laion_clap not installed — CLAP scoring unavailable.")

try:
    import jiwer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False

try:
    from transformers import pipeline as hf_pipeline
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    warnings.warn("transformers not installed — Whisper ASR unavailable.")

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False


DOMAIN_WEIGHTS: Dict[str, tuple] = {
    "speech":    (0.10, 0.90),
    "speecht5":  (0.10, 0.90),
    "bark":      (0.15, 0.85),
    "tts":       (0.10, 0.90),
    "audioldm":  (1.00, 0.00),
    "audioldm2": (1.00, 0.00),
    "musicgen":  (1.00, 0.00),
    "music":     (1.00, 0.00),
    "ambient":   (1.00, 0.00),
    "sfx":       (1.00, 0.00),
}

_SPEECH_DOMAINS = {"speech", "speecht5", "bark", "tts"}


class AudioEvaluator:
    """
    Unified evaluator for text-to-audio, text-to-music, and TTS models.

    Fix log (v2)
    ------------
    1. Whisper diagnostic logging — empty transcripts are now visible.
    2. WAV format guard — float32 files are flagged before Whisper sees them.
    3. asr_reference parameter — separates the CLAP query from the WER reference.
    4. No mock scores, no silent fallback (unchanged from v1).
    """

    def __init__(self, use_cuda: bool = True, whisper_model: str = "base.en"):
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.clap_model = None
        self._asr = None
        self._whisper_model_name = whisper_model

        if HAS_CLAP:
            self._load_clap()

    def _load_clap(self):
        """
        Try HTSAT-tiny first (lighter), then HTSAT-base.
        Raises RuntimeError if both fail — no silent mock fallback.
        """
        attempts = [
            dict(enable_fusion=False, amodel="HTSAT-tiny"),
            dict(enable_fusion=True,  amodel="HTSAT-base"),
        ]
        for kwargs in attempts:
            try:
                model = laion_clap.CLAP_Module(**kwargs)
                model.load_ckpt()
                self.clap_model = model
                print(f"[CLAP] Loaded {kwargs['amodel']} "
                      f"(fusion={kwargs['enable_fusion']}) on {self.device}")
                return
            except Exception as exc:
                print(f"[CLAP] {kwargs['amodel']} failed: {exc}")

        raise RuntimeError(
            "[CLAP] All checkpoint attempts failed. "
            "Check GPU memory and network access to Hugging Face. "
            "The server will not start until CLAP loads successfully."
        )

    # ── WAV format guard ──────────────────────────────────────────────────────

    def _check_wav_format(self, audio_path: str) -> bool:
        """
        Return True if the WAV file is in a format Whisper can read (int16 PCM).
        Logs a warning if the file appears to be float32 — the most common
        cause of silent empty transcripts from SpeechT5-generated audio.
        """
        if not HAS_SF:
            return True   # can't check without soundfile, proceed anyway
        try:
            info = sf.info(audio_path)
            if info.subtype not in ("PCM_16", "PCM_24", "PCM_32"):
                warnings.warn(
                    f"[Whisper] {audio_path} has subtype '{info.subtype}'. "
                    f"Whisper expects PCM_16. "
                    f"Fix: save with sf.write(path, audio_int16, 16000, subtype='PCM_16'). "
                    f"Transcript may be empty."
                )
                return False
        except Exception as exc:
            warnings.warn(f"[Whisper] Could not inspect {audio_path}: {exc}")
        return True

    # ── Whisper ASR (lazy, speech domains only) ───────────────────────────────

    def _get_asr(self):
        if self._asr is not None:
            return self._asr
        if not HAS_WHISPER:
            return None
        try:
            self._asr = hf_pipeline(
                "automatic-speech-recognition",
                model=f"openai/whisper-{self._whisper_model_name}",
                device=0 if self.device == "cuda" else -1,
            )
            print(f"[Whisper] Loaded whisper-{self._whisper_model_name}")
        except Exception as exc:
            warnings.warn(f"[Whisper] Load failed: {exc}. Semantic score will be 0.")
            self._asr = None
        return self._asr

    def _transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio_path with Whisper.
        Logs a warning if the transcript is empty so the operator can act.
        """
        # Format check before handing to Whisper
        self._check_wav_format(audio_path)

        asr = self._get_asr()
        if asr is None:
            warnings.warn("[Whisper] ASR pipeline not available — semantic_score will be 0.")
            return ""
        try:
            result     = asr(audio_path, return_timestamps=False)
            transcript = result.get("text", "").strip()

            if not transcript:
                warnings.warn(
                    f"[Whisper] Empty transcript for: {audio_path}\n"
                    f"  Possible causes: float32 WAV, silence, or very short audio.\n"
                    f"  semantic_score will be 0 for this sample."
                )
            else:
                print(f"[Whisper] Transcript: {transcript!r}")

            return transcript
        except Exception as exc:
            warnings.warn(f"[Whisper] Transcription error for {audio_path}: {exc}")
            return ""

    # ── CLAP similarity ───────────────────────────────────────────────────────

    def evaluate_clap(self, text_prompt: str, audio_path: str) -> float:
        """
        Real CLAP cosine similarity in [0, 1].
        Raises RuntimeError if called before CLAP loaded successfully.
        No mock fallback — a fake score is worse than a visible error.
        """
        if self.clap_model is None:
            raise RuntimeError(
                "[CLAP] Model not loaded. Check server startup logs."
            )
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(
                f"[CLAP] Audio file not found: {audio_path}"
            )

        # CLAP truncation fix
        # -------------------
        # laion_clap uses a BERT/CLIP-style tokenizer with model_max_length=77.
        # get_text_embedding() calls the tokenizer internally with no explicit
        # max_length or truncation argument, so any prompt longer than 77 tokens
        # is silently truncated and the tail is dropped.
        #
        # For RAG-on runs the enhanced prompt can easily be 80-150 tokens, meaning
        # the most specific descriptors appended by the RAG system (which appear
        # at the end) are the first tokens to be cut. This is the opposite of what
        # we want — the original prompt is at the front and is fine; the KB context
        # is at the back and gets silently dropped.
        #
        # Fix: truncate the text to 77 CLAP tokens ourselves, before calling
        # get_text_embedding(), so the truncation is:
        #   a) deliberate and visible (we log it),
        #   b) consistent — same tokens every time, not dependent on internal
        #      batching order inside laion_clap.
        #
        # We use the model's own tokenizer so the token count is exact.
        # If the tokenizer isn't exposed we fall back to a word-level heuristic
        # (77 tokens ≈ 55-60 English words for this vocabulary).

        _CLAP_MAX_TOKENS = 77

        clap_prompt = text_prompt
        try:
            # laion_clap exposes its tokenizer via model.tokenizer or model.text_branch
            _tokenizer = getattr(self.clap_model, "tokenizer", None)
            if _tokenizer is None:
                _tokenizer = getattr(
                    getattr(self.clap_model, "text_branch", None), "tokenizer", None
                )

            if _tokenizer is not None:
                tokens = _tokenizer(
                    text_prompt,
                    truncation=True,
                    max_length=_CLAP_MAX_TOKENS,
                    return_tensors="pt",
                )
                # Decode back to a string so get_text_embedding receives clean text
                clap_prompt = _tokenizer.decode(
                    tokens["input_ids"][0],
                    skip_special_tokens=True,
                )
                original_token_count = len(_tokenizer(text_prompt)["input_ids"])
                if original_token_count > _CLAP_MAX_TOKENS:
                    print(
                        f"[CLAP] Prompt was {original_token_count} tokens — "
                        f"truncated to {_CLAP_MAX_TOKENS} for CLAP embedding.\n"
                        f"  Original : {text_prompt[:100]!r}…\n"
                        f"  Truncated: {clap_prompt[:100]!r}"
                    )
            else:
                # Fallback: word-level heuristic (≈ 0.75 words per token for this vocab)
                words = text_prompt.split()
                _word_limit = int(_CLAP_MAX_TOKENS * 0.75)
                if len(words) > _word_limit:
                    clap_prompt = " ".join(words[:_word_limit])
                    print(
                        f"[CLAP] Prompt heuristically truncated to {_word_limit} words "
                        f"(tokenizer not accessible).\n"
                        f"  Original : {text_prompt[:100]!r}…\n"
                        f"  Truncated: {clap_prompt[:100]!r}"
                    )
        except Exception as _trunc_exc:
            print(f"[CLAP] Truncation pre-check failed ({_trunc_exc}); using full prompt.")
            clap_prompt = text_prompt

        text_embed  = self.clap_model.get_text_embedding([clap_prompt])
        audio_embed = self.clap_model.get_audio_embedding_from_filelist(
            x=[audio_path], use_tensor=False
        )

        text_t  = torch.tensor(text_embed,  dtype=torch.float32)
        audio_t = torch.tensor(audio_embed, dtype=torch.float32)

        raw = torch.nn.functional.cosine_similarity(
            text_t, audio_t, dim=-1
        ).mean().item()

        score = round(float(max(0.0, min(1.0, raw))), 4)
        print(f"[CLAP] score={score:.4f} | {text_prompt[:70]!r}")
        return score

    # ── Semantic / WER ────────────────────────────────────────────────────────

    def text_similarity(self, reference: str, hypothesis: str) -> float:
        if not hypothesis:
            return 0.0

        if HAS_JIWER:
            wer   = jiwer.wer(reference.lower().strip(), hypothesis.lower().strip())
            score = round(max(0.0, 1.0 - wer), 4)
            print(f"[Semantic] WER={wer:.3f} → similarity={score:.4f}")
            return score

        ref_words = set(reference.lower().split())
        hyp_words = set(hypothesis.lower().split())
        if not ref_words:
            return 0.0
        union = ref_words | hyp_words
        score = round(len(ref_words & hyp_words) / len(union), 4)
        print(f"[Semantic] Jaccard={score:.4f}")
        return score

    # ── Unified POAS — use this from app.py / batch_eval.py ──────────────────

    def evaluate_poas(
        self,
        text_prompt: str,
        audio_path: str,
        model_type: str = "audioldm",
        domain: Optional[str] = None,
        asr_text: Optional[str] = None,
        asr_reference: Optional[str] = None,
        inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Returns poas_score plus all sub-scores.

        Parameters
        ----------
        text_prompt    : The prompt sent to the audio model (possibly RAG-enhanced).
                         This is what CLAP compares against the audio.
        audio_path     : Path to the generated WAV file.
        model_type     : Used to look up DOMAIN_WEIGHTS routing key.
        domain         : Fallback routing key if model_type not in DOMAIN_WEIGHTS.
        asr_text       : Pre-computed transcript (skips Whisper if provided).
        asr_reference  : The *original* (non-enhanced) prompt to compare the
                         Whisper transcript against for WER calculation.
                         If None, text_prompt is used (backward-compatible).
                         For RAG-on runs, pass the original prompt here so WER
                         reflects what was actually spoken, not the long RAG string.
        inference_steps: Logged for reference only.
        seed           : Logged for reference only.

        Scoring
        -------
        Speech domains  :  0.10 × clap_score + 0.90 × semantic_score
        All other domains: 1.00 × clap_score
        """
        clap_score = self.evaluate_clap(text_prompt, audio_path)

        route_key = (model_type or "").lower()
        if route_key not in DOMAIN_WEIGHTS and domain:
            route_key = domain.lower()

        clap_w, sem_w = DOMAIN_WEIGHTS.get(route_key, (1.00, 0.00))

        semantic_score  = 0.0
        transcript_used = ""

        if sem_w > 0:
            # Determine what was actually spoken (original prompt, not RAG-extended)
            wer_reference = asr_reference if asr_reference else text_prompt

            if asr_text:
                transcript_used = asr_text
            elif route_key in _SPEECH_DOMAINS and os.path.isfile(audio_path):
                transcript_used = self._transcribe(audio_path)

            if not transcript_used:
                # Keep semantic_score = 0 but make the zero explicitly visible
                print(
                    f"[POAS] WARNING: transcript is empty for {audio_path!r}. "
                    f"semantic_score forced to 0.0. "
                    f"Check Whisper load status and WAV format."
                )
            else:
                semantic_score = self.text_similarity(wer_reference, transcript_used)

        raw  = clap_w * clap_score + sem_w * semantic_score
        poas = round(float(max(0.0, min(1.0, raw))), 4)

        print(f"[POAS] {route_key}: "
              f"clap={clap_score:.4f}×{clap_w} + "
              f"sem={semantic_score:.4f}×{sem_w} = {poas:.4f}")

        return {
            "poas_score":      poas,
            "clap_score":      clap_score,
            "semantic_score":  round(semantic_score, 4),
            "clap_weight":     clap_w,
            "semantic_weight": sem_w,
            "transcript":      transcript_used,
            "inference_steps": inference_steps,
            "seed":            seed,
        }

    # ── Batch evaluation ──────────────────────────────────────────────────────

    def evaluate_batch(self, samples: List[Dict]) -> List[Dict]:
        results = []
        for item in samples:
            scores = self.evaluate_poas(
                text_prompt=item["prompt"],
                audio_path=item["audio_path"],
                model_type=item.get("model_type", "audioldm"),
                domain=item.get("domain"),
                asr_text=item.get("asr_text"),
                asr_reference=item.get("original_prompt"),   # NEW: pass original for WER
                inference_steps=item.get("inference_steps"),
                seed=item.get("seed"),
            )
            results.append({
                "domain":     item.get("domain", "unknown"),
                "model":      item.get("model_type", "unknown"),
                "prompt":     item["prompt"],
                "audio_path": item["audio_path"],
                **scores,
            })
        return results

    # ── Aggregate metrics ─────────────────────────────────────────────────────

    def compute_aggregate_metrics(self, results_data: List[Dict]) -> Dict:
        if not results_data:
            return {}

        domain_scores: Dict[str, List[float]] = collections.defaultdict(list)
        model_scores:  Dict[str, List[float]] = collections.defaultdict(list)
        all_scores: List[float] = []

        for item in results_data:
            score = item["poas_score"]
            domain_scores[item["domain"]].append(score)
            model_scores[item["model"]].append(score)
            all_scores.append(score)

        domain_stds = [float(np.std(v)) for v in domain_scores.values() if len(v) > 1]
        cri = round(float(np.mean(domain_stds)) if domain_stds else 0.0, 4)

        return {
            "POAS_Global_Mean":       round(float(np.mean(all_scores)), 4),
            "CRI_Domain_Consistency": cri,
            "Diversity_Index":        round(float(np.std(all_scores)), 4),
            "Model_Means":            {k: round(float(np.mean(v)), 4) for k, v in model_scores.items()},
            "Domain_Summary":         {k: {"mean": round(float(np.mean(v)), 4),
                                           "std":  round(float(np.std(v)),  4),
                                           "n":    len(v)}
                                       for k, v in domain_scores.items()},
        }

    def export_results(self, results_data: List[Dict], output_csv: str) -> pd.DataFrame:
        df = pd.DataFrame(results_data)
        df.to_csv(output_csv, index=False)
        return df
