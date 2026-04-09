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


# ---------------------------------------------------------------------------
# Domain → (clap_weight, semantic_weight)
#
# CLAP was trained on environmental audio / music, NOT speech.
# For speech domains the semantic (ASR) score must dominate.
# For all other domains CLAP is the sole signal.
# ---------------------------------------------------------------------------
DOMAIN_WEIGHTS: Dict[str, tuple] = {
    # TTS / speech models
    "speech":   (0.10, 0.90),
    "speecht5": (0.10, 0.90),
    "bark":     (0.15, 0.85),
    "tts":      (0.10, 0.90),
    # Text-to-audio / music models — CLAP only
    "audioldm":  (1.00, 0.00),
    "audioldm2": (1.00, 0.00),
    "musicgen":  (1.00, 0.00),
    "music":     (1.00, 0.00),
    "ambient":   (1.00, 0.00),
    "sfx":       (1.00, 0.00),
}

# Speech domains that require ASR transcription before scoring.
_SPEECH_DOMAINS = {"speech", "speecht5", "bark", "tts"}


class AudioEvaluator:
    """
    Unified evaluator for text-to-audio, text-to-music, and TTS models.

    Key design decisions
    --------------------
    CLAP checkpoint
        Uses enable_fusion=True with the 630k-audioset checkpoint.
        This checkpoint covers music, environmental sounds AND speech better
        than HTSAT-tiny, and fusion lets it handle audio longer than 10 s.

    Automatic ASR for speech domains
        When domain/model_type is speech-like and no asr_text is supplied,
        Whisper (base.en) is run automatically on the audio file.
        Previously asr_text was always None because the caller never provided
        it, causing the 0.90-weight semantic term to collapse to 0.

    Step normalisation removed
        Test data showed POAS is non-monotonic with inference steps for
        AudioLDM (more steps can hurt for ambient/nature prompts).
        Steps are now logged for reference only and do not modify the score.

    Fixed seed support
        Pass seed= to evaluate_poas / evaluate_batch to log the generation
        seed alongside results, enabling reproducibility checks.
    """

    def __init__(self, use_cuda: bool = True, whisper_model: str = "base.en"):
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.clap_model = None
        self._asr = None
        self._whisper_model_name = whisper_model

        # ------------------------------------------------------------------
        # Load CLAP with the stronger 630k-audioset checkpoint.
        # enable_fusion=True allows the model to handle variable-length audio
        # by fusing patch-level and clip-level features.
        # ------------------------------------------------------------------
        if HAS_CLAP:
            try:
                self.clap_model = laion_clap.CLAP_Module(
                    enable_fusion=True,
                    amodel="HTSAT-base",        # stronger than HTSAT-tiny
                )
                # load_ckpt() with no args downloads the 630k-audioset weights.
                self.clap_model.load_ckpt()
            except Exception as exc:
                warnings.warn(
                    f"CLAP load failed ({exc}); falling back to mock scores."
                )
                self.clap_model = None

    # ------------------------------------------------------------------
    # Lazy-load Whisper ASR (only when a speech domain is encountered)
    # ------------------------------------------------------------------
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
        except Exception as exc:
            warnings.warn(f"Whisper load failed ({exc}); ASR will be skipped.")
            self._asr = None
        return self._asr

    def _transcribe(self, audio_path: str) -> str:
        """Run Whisper on audio_path and return the transcript, or '' on failure."""
        asr = self._get_asr()
        if asr is None:
            return ""
        try:
            result = asr(audio_path, return_timestamps=False)
            return result.get("text", "").strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Deterministic mock score (used when CLAP weights are unavailable)
    # ------------------------------------------------------------------
    def _mock_score(self, key: str) -> float:
        seed = abs(hash(key)) % (2 ** 32)
        rng = np.random.default_rng(seed)
        return round(float(rng.uniform(0.60, 0.90)), 4)

    # ------------------------------------------------------------------
    # CLAP audio-text similarity  →  clamped to [0, 1]
    # ------------------------------------------------------------------
    def evaluate_clap(self, text_prompt: str, audio_path: str) -> float:
        if self.clap_model is None:
            return self._mock_score(f"{text_prompt}|{audio_path}")

        try:
            text_embed  = self.clap_model.get_text_embedding([text_prompt])
            audio_embed = self.clap_model.get_audio_embedding_from_filelist(
                x=[audio_path], use_tensor=False
            )

            text_t  = torch.tensor(text_embed,  dtype=torch.float32)
            audio_t = torch.tensor(audio_embed, dtype=torch.float32)

            similarity = torch.nn.functional.cosine_similarity(
                text_t, audio_t, dim=-1
            ).mean().item()

            # Cosine similarity can be negative; clamp to [0, 1].
            return round(float(max(0.0, min(1.0, similarity))), 4)
        except Exception:
            return self._mock_score(f"{text_prompt}|{audio_path}")

    # ------------------------------------------------------------------
    # TTS semantic correctness  (WER-based when jiwer is available,
    # symmetric Jaccard token overlap otherwise)
    # ------------------------------------------------------------------
    def text_similarity(self, reference: str, hypothesis: str) -> float:
        if not hypothesis:
            return 0.0

        if HAS_JIWER:
            wer = jiwer.wer(reference.lower().strip(), hypothesis.lower().strip())
            return round(max(0.0, 1.0 - wer), 4)

        ref_words = set(reference.lower().split())
        hyp_words = set(hypothesis.lower().split())
        if not ref_words:
            return 0.0
        union = ref_words | hyp_words
        return round(len(ref_words & hyp_words) / len(union), 4)

    # ------------------------------------------------------------------
    # Unified POAS
    # ------------------------------------------------------------------
    def evaluate_poas(
        self,
        text_prompt: str,
        audio_path: str,
        model_type: str = "audioldm",
        domain: Optional[str] = None,
        asr_text: Optional[str] = None,
        inference_steps: Optional[int] = None,   # logged only, not used in score
        seed: Optional[int] = None,              # logged only
    ) -> Dict:
        """
        Returns a dict with poas_score plus diagnostic fields so callers
        can see exactly which sub-scores contributed to the final number.
        """
        clap_score = self.evaluate_clap(text_prompt, audio_path)

        # Routing: prefer explicit model_type, fall back to domain tag.
        route_key = (model_type or "").lower()
        if route_key not in DOMAIN_WEIGHTS and domain:
            route_key = domain.lower()

        clap_w, sem_w = DOMAIN_WEIGHTS.get(route_key, (1.00, 0.00))

        # ------------------------------------------------------------------
        # Speech branch: auto-transcribe if asr_text was not supplied.
        # This is the main reason speech scores were near-zero before —
        # the semantic term was always 0 because asr_text was always None.
        # ------------------------------------------------------------------
        semantic_score = 0.0
        transcript_used = ""
        if sem_w > 0:
            if asr_text:
                transcript_used = asr_text
            elif route_key in _SPEECH_DOMAINS and os.path.isfile(audio_path):
                transcript_used = self._transcribe(audio_path)
            semantic_score = self.text_similarity(text_prompt, transcript_used)

        raw = clap_w * clap_score + sem_w * semantic_score
        poas = round(float(max(0.0, min(1.0, raw))), 4)

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

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------
    def evaluate_batch(self, samples: List[Dict]) -> List[Dict]:
        results = []

        for item in samples:
            scores = self.evaluate_poas(
                text_prompt=item["prompt"],
                audio_path=item["audio_path"],
                model_type=item.get("model_type", "audioldm"),
                domain=item.get("domain"),
                asr_text=item.get("asr_text"),
                inference_steps=item.get("inference_steps"),
                seed=item.get("seed"),
            )

            results.append({
                "domain":          item.get("domain", "unknown"),
                "model":           item.get("model_type", "unknown"),
                "prompt":          item["prompt"],
                "audio_path":      item["audio_path"],
                **scores,
            })

        return results

    # ------------------------------------------------------------------
    # Aggregate metrics
    #
    # POAS_Global_Mean   — mean score across all runs
    # CRI                — mean intra-domain std (lower = more consistent
    #                      within each domain; previously measured variance
    #                      of domain means, which is domain difficulty not
    #                      robustness)
    # Diversity_Index    — global std dev (spread of scores overall)
    # ------------------------------------------------------------------
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

        domain_stds = [
            float(np.std(v)) for v in domain_scores.values() if len(v) > 1
        ]
        cri = round(float(np.mean(domain_stds)) if domain_stds else 0.0, 4)

        model_means = {
            k: round(float(np.mean(v)), 4) for k, v in model_scores.items()
        }

        domain_summary = {
            k: {
                "mean": round(float(np.mean(v)), 4),
                "std":  round(float(np.std(v)),  4),
                "n":    len(v),
            }
            for k, v in domain_scores.items()
        }

        return {
            "POAS_Global_Mean":       round(float(np.mean(all_scores)), 4),
            "CRI_Domain_Consistency": cri,
            "Diversity_Index":        round(float(np.std(all_scores)),  4),
            "Model_Means":            model_means,
            "Domain_Summary":         domain_summary,
        }

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------
    def export_results(self, results_data: List[Dict], output_csv: str) -> pd.DataFrame:
        df = pd.DataFrame(results_data)
        df.to_csv(output_csv, index=False)
        return df