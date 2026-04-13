"""
rag_enhancer.py  (v2 — improved)
=================================
Key changes vs v1
-----------------
1. Whole-word key matching
   The old code used `if key in prompt_lower` which matched substrings.
   "door" matched inside "outdoor", "dog" partially overlapped "doing", etc.
   Now we use a regex word-boundary check so every key must appear as a
   whole word (or whole phrase for multi-word keys).

2. Context deduplication
   Multiple KB keys can expand to the same or very similar text
   (e.g. "storm" and "thunder" both expand to the same thunderstorm block).
   We now fingerprint each retrieved context and skip duplicates.

3. Hard cap on retrieved context length
   CLAP was trained on short AudioCaps captions (~10-20 words).
   Very long enhanced prompts dilute the text embedding and reduce cosine
   similarity — as seen in the piano+rain prompt (120 words → POAS 0.22).
   We now cap total retrieved context to MAX_CONTEXT_WORDS (default 20).

4. Smarter first-clause extraction
   Instead of dumping the whole KB entry we take only the first descriptive
   clause (up to the first comma), keeping the most salient descriptor.
   This also helps stay within the word cap.

5. Speech domain unchanged (intentional)
   SpeechT5 is sensitive to added descriptors in the TTS text.
   Speech prompts are passed through without any RAG augmentation.
"""

import re


MAX_CONTEXT_WORDS = 20   # hard cap on words added to the prompt from KB


class PromptEnhancer:
    def __init__(self):
        """
        Initializes a mock Local Knowledge Base (KB) for Prompt Enhancement (RAG).
        In a full enterprise project, this would integrate with a vector database
        (e.g., Chroma, FAISS) and an LLM API. Due to API limitations for this
        mini-project, we use a structured local KB to simulate
        retrieval-augmented context injection.
        """
        self.knowledge_base = {

            # ── Instruments ───────────────────────────────────────────────
            "guitar": (
                "acoustic guitar playing, warm and resonant string sound, "
                "finger-picked melody, natural wood body tone, "
                "gentle plucking and strumming"
            ),
            "electric guitar": (
                "electric guitar with distortion or clean tone, "
                "rock or blues guitar sound, sustained notes and chords, "
                "amplifier buzz, expressive bends and vibrato"
            ),
            "piano": (
                "piano playing with rich full tone, "
                "keys struck with feeling, sustained notes resonating, "
                "warm and expressive piano melody, acoustic grand piano sound"
            ),
            "violin": (
                "violin playing with a bow, singing string melody, "
                "warm and emotional tone, expressive vibrato, "
                "classical or folk violin sound"
            ),
            "drums": (
                "drum kit playing, kick drum and snare, "
                "cymbals crashing, rhythmic and energetic, "
                "live acoustic drumming"
            ),
            "bass": (
                "bass guitar playing, deep low tones, "
                "groovy rhythmic bass line, warm and full low-end, "
                "finger-plucked or slapped bass"
            ),
            "trumpet": (
                "trumpet playing bright and bold, "
                "brass fanfare or jazz melody, "
                "expressive and powerful trumpet sound"
            ),
            "flute": (
                "flute playing a gentle melody, "
                "airy and light tone, breathy and delicate, "
                "classical or folk flute sound"
            ),
            "synthesizer": (
                "synthesizer producing electronic tones, "
                "lush pads or sharp leads, electronic music texture, "
                "futuristic and evolving synth sound"
            ),

            # ── Music genres ──────────────────────────────────────────────
            "techno": (
                "fast electronic dance music, heavy repetitive kick drum beat, "
                "deep bass, mechanical rhythmic pattern, dark industrial atmosphere, "
                "pulsing synthesizer, energetic and hypnotic"
            ),
            "jazz": (
                "relaxed jazz ensemble, swinging rhythm, upright bass walking melody, "
                "soft brushed drums, piano chords, saxophone or trumpet improvisation, "
                "warm intimate atmosphere, smoky club feeling"
            ),
            "classical": (
                "full orchestra playing together, grand concert hall, "
                "strings and brass and woodwinds, powerful and expressive, "
                "quiet passages and loud climaxes, elegant and dramatic"
            ),
            "hip hop": (
                "hip hop beat with heavy bass, sampled drums, rhythmic groove, "
                "urban street music, punchy kick and snare, "
                "looping instrumental, energetic and confident"
            ),
            "ambient": (
                "slow peaceful background music, soft evolving tones, "
                "calm and relaxing atmosphere, gentle and spacious, "
                "meditative and soothing, no strong rhythm"
            ),
            "pop": (
                "bright catchy pop song, upbeat melody, "
                "clear vocals and chorus, energetic and polished, "
                "modern radio-friendly production"
            ),
            "folk": (
                "folk music with acoustic instruments, natural and warm, "
                "storytelling feel, simple and heartfelt, "
                "acoustic guitar or banjo, gentle and rustic"
            ),
            "orchestral": (
                "full orchestra, sweeping strings and brass, "
                "grand cinematic sound, powerful dynamics, "
                "dramatic and emotionally expressive"
            ),
            "808": (
                "deep booming 808 bass, sub-bass kick, "
                "electronic drum machine, punchy and heavy low end"
            ),

            # ── Nature & weather ──────────────────────────────────────────
            "storm": (
                "loud thunderstorm, heavy rain falling, "
                "thunder rumbling in the distance, lightning cracking, "
                "strong wind gusting, dark and dramatic outdoor weather"
            ),
            "thunder": (
                "loud thunderclap cracking overhead, "
                "deep rumbling boom, dramatic storm atmosphere"
            ),
            "rain": (
                "rain falling steadily, water droplets hitting leaves and ground, "
                "gentle patter becoming heavier, puddles forming, "
                "soft and peaceful outdoor rain sound"
            ),
            "wind": (
                "strong wind blowing outdoors, gusts through trees, "
                "leaves rustling, breezy and open environment, "
                "howling wind in the distance"
            ),
            "ocean": (
                "ocean waves crashing on the shore, water rushing up the beach, "
                "seagulls calling overhead, gentle sea breeze, "
                "rhythmic and calming seaside atmosphere"
            ),
            "fire": (
                "wood crackling in a fire, flames burning steadily, "
                "occasional pops and sparks, warm and cosy fireplace sound, "
                "gentle roar of an open fire"
            ),
            "forest": (
                "peaceful forest with birds singing, leaves rustling in the breeze, "
                "distant stream babbling, insects humming quietly, "
                "calm and natural outdoor woodland"
            ),
            "river": (
                "river flowing over rocks, water rushing and bubbling, "
                "gentle stream in a quiet valley, "
                "soothing continuous water sound outdoors"
            ),

            # ── Birds & animals ───────────────────────────────────────────
            "bird": (
                "birds singing outdoors, chirping and trilling in the trees, "
                "dawn chorus in a quiet garden or woodland, "
                "peaceful natural bird sounds, multiple birds calling"
            ),
            "dog": (
                "dog barking loudly, growling and panting, "
                "playful or alert dog sounds, collar jingling, "
                "indoor or outdoor dog noise"
            ),
            "crowd": (
                "large crowd of people talking and cheering, "
                "applause and chatter, busy and lively atmosphere, "
                "audience noise in a big venue"
            ),

            # ── Urban & indoor environments ───────────────────────────────
            "restaurant": (
                "busy restaurant with people talking and eating, "
                "cutlery clinking on plates, background music playing softly, "
                "kitchen sounds in the distance, lively dining atmosphere"
            ),
            "cafe": (
                "quiet coffee shop, espresso machine running, "
                "soft background music, people chatting quietly, "
                "cups and saucers clinking, relaxed and cosy atmosphere"
            ),
            "coffee": (
                "quiet coffee shop, espresso machine running, "
                "soft background music, people chatting quietly, "
                "cups and saucers clinking, relaxed and cosy atmosphere"
            ),
            "city": (
                "busy city street with cars and buses passing, "
                "people walking and talking on the pavement, "
                "traffic noise and distant sirens, urban outdoor environment"
            ),
            "subway": (
                "underground train arriving at a station, "
                "rails screeching, doors opening and closing, "
                "crowd of commuters, announcements over the speakers, "
                "echoing tunnel noise"
            ),
            "office": (
                "quiet office environment, keyboards typing, "
                "air conditioning humming, phone ringing in the distance, "
                "people working, calm indoor workplace sounds"
            ),
            "intersection": (
                "busy road intersection, cars stopping and going, "
                "traffic signals, pedestrians crossing, "
                "urban street sounds and vehicle noise"
            ),

            # ── Speech & voice ────────────────────────────────────────────
            "speech": (
                "clear human voice speaking, good diction, "
                "clean and quiet recording, natural speaking tone"
            ),
            "whisper": (
                "person whispering quietly, soft and breathy voice, "
                "very close and intimate, barely audible"
            ),
            "narration": (
                "narrator speaking clearly and calmly, "
                "warm and authoritative voice, steady pacing, "
                "documentary or audiobook style"
            ),

            # ── Mechanical & everyday SFX ─────────────────────────────────
            "footsteps": (
                "footsteps walking on a surface, "
                "shoes hitting the ground rhythmically, "
                "walking on gravel or wood or concrete"
            ),
            "door": (
                "door opening or closing, handle clicking, "
                "hinges creaking, wooden door sound"
            ),
            "explosion": (
                "loud explosion with deep rumble, "
                "blast and shockwave, debris falling, "
                "dramatic booming sound fading away"
            ),
            "engine": (
                "engine running and revving, "
                "motor rumbling steadily, "
                "car or machine engine noise"
            ),
            "clock": (
                "clock ticking steadily, "
                "second hand moving, mechanical clockwork sound, "
                "regular rhythmic ticking"
            ),
            "keys": (
                "keys jingling and clinking together, "
                "small metal objects rattling, "
                "light and delicate metallic sound"
            ),
            "fireplace": (
                "wood crackling in a fireplace, flames popping, "
                "warm cosy indoor fire sounds, "
                "occasional spark and sizzle"
            ),
        }

        # Precompile whole-word regex patterns for each key
        # Multi-word keys (e.g. "electric guitar") use a phrase match.
        self._key_patterns = {
            key: re.compile(r"\b" + re.escape(key) + r"\b", re.IGNORECASE)
            for key in self.knowledge_base
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _first_clause(self, context: str) -> str:
        """
        Return the first comma-delimited clause of a KB entry.
        This is the most salient descriptor and keeps prompts short.
        """
        return context.split(",")[0].strip()

    def fetch_retrieved_context(
        self,
        prompt: str,
        max_words: int = MAX_CONTEXT_WORDS,
    ) -> str:
        """
        Retrieve KB entries whose keys appear as whole words in the prompt.

        Improvements over v1:
        - Whole-word regex matching (no substring false-positives)
        - Deduplication of identical context fingerprints
        - Hard cap on total output word count
        """
        seen_fingerprints: set = set()
        retrieved_clauses: list = []

        # Sort keys by length descending so longer / more specific keys
        # (e.g. "electric guitar") are checked before shorter ones ("guitar").
        sorted_keys = sorted(self.knowledge_base.keys(), key=len, reverse=True)

        for key in sorted_keys:
            if not self._key_patterns[key].search(prompt):
                continue

            context = self.knowledge_base[key]
            clause  = self._first_clause(context)

            # Deduplicate by the first 40 characters of the full entry
            fingerprint = context[:40]
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)

            retrieved_clauses.append(clause)

        if not retrieved_clauses:
            return "clear and realistic sound, good quality audio"

        combined = ", ".join(retrieved_clauses)

        # Hard word-count cap — keeps CLAP embedding focused
        words = combined.split()
        if len(words) > max_words:
            combined = " ".join(words[:max_words])

        return combined

    # ── Public enhance() ──────────────────────────────────────────────────────

    def enhance(self, prompt: str, domain: str) -> str:
        """
        Build an enhanced prompt for the given domain.

        Domain routing
        --------------
        music   : "High quality music, {prompt}, {context}"
        sfx     : "Sound effect of {prompt}, {context}"
        ambient : "Ambient sound of {prompt}, {context}"
        speech  : pass-through — SpeechT5 is sensitive to appended descriptors
        other   : "{prompt}, {context}"
        """
        if domain == "speech":
            # Speech: intentionally no RAG augmentation.
            # SpeechT5 treats the entire string as text to speak aloud;
            # appending audio-description phrases would be spoken verbatim.
            return prompt

        retrieved = self.fetch_retrieved_context(prompt)

        if domain == "music":
            return f"High quality music, {prompt}, {retrieved}"
        elif domain == "sfx":
            return f"Sound effect of {prompt}, {retrieved}"
        elif domain == "ambient":
            return f"Ambient sound of {prompt}, {retrieved}"
        else:
            return f"{prompt}, {retrieved}"


if __name__ == "__main__":
    # Quick smoke-test
    enhancer = PromptEnhancer()

    test_cases = [
        ("A melancholic jazz piano trio in a dimly lit club", "music"),
        ("Soft piano melody over gentle rain in a quiet room", "music"),
        ("Footsteps walking on gravel at a steady pace", "sfx"),
        ("Keys jingling and a door handle clicking open", "sfx"),
        ("A dog barking repeatedly in an outdoor yard", "sfx"),
        ("A cosy coffee shop with soft chatter and an espresso machine", "ambient"),
        ("Heavy rain falling on a rooftop during a thunderstorm", "ambient"),
        ("Welcome to the annual science fair. Please find your seats.", "speech"),
    ]

    print("=" * 70)
    for prompt, domain in test_cases:
        enhanced = enhancer.enhance(prompt, domain)
        word_count = len(enhanced.split())
        print(f"[{domain:7s}] ({word_count:3d} words) {enhanced[:100]}")
    print("=" * 70)
    print("PromptEnhancer v2 loaded successfully.")
