import random

class PromptEnhancer:
    def __init__(self):
        """
        Initializes a mock Local Knowledge Base (KB) for Prompt Enhancement (RAG).
        In a full enterprise project, this would integrate with a vector database (e.g., Chroma, FAISS)
        and an LLM API. Due to API limitations for this mini-project, we use a structured local KB
        to simulate retrieval-augmented context injection.
        """
        self.knowledge_base = {
            "guitar": "high quality, crisp acoustic resonance, rich harmonics, 44.1kHz standard studio recording",
            "piano": "grand piano, sustained pedal, classical reverb, high fidelity audio",
            "techno": "128 bpm, sidechain compression, 808 kick, sub-bass frequencies, electronic dynamic range",
            "speech": "close-mic, voice-over quality, clear diction, noise-free, isolated vocal track",
            "storm": "low frequency rumble, binaural recording, immersive environmental audio surround",
            "restaurant": "ambient field recording, medium stereo width, dynamic chatter, distinct clinking highs",
            "bird": "outdoor field recording, high frequency trills, natural ambiance, wide stereo"
        }

    def fetch_retrieved_context(self, prompt: str) -> str:
        """
        Simulates retrieving chunks from a vector database based on prompt keywords.
        """
        prompt_lower = prompt.lower()
        retrieved_pieces = []
        for key, context in self.knowledge_base.items():
            if key in prompt_lower:
                retrieved_pieces.append(context)
        
        if retrieved_pieces:
            # Combine retrieved contexts
            return ", ".join(retrieved_pieces)
        
        # Default fallback context for overall audio enhancement
        return "high quality, clear audio, realistic soundscape, lossless format"

    def enhance(self, prompt, domain):
        if domain == "music":
            return f"High quality cinematic music, {prompt}, rich stereo mix"
        elif domain == "speech":
            return f"{prompt}"
        elif domain == "sfx":
            return f"Realistic sound effect, clean isolated audio, {prompt}"
        elif domain == "ambient":
            return f"Immersive ambient soundscape, wide stereo atmosphere, natural environmental depth, realistic background ambience, {prompt}"


if __name__ == "__main__":
    enhancer = PromptEnhancer()
    test_prompt = "A cheerful pop song with a fast tempo and bright piano."
    print("Original:", test_prompt)
    print("Enhanced:", enhancer.enhance(test_prompt))
