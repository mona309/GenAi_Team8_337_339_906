import random

class PromptEnhancer:
    def __init__(self):
        """
        Local Knowledge Base for RAG-based Prompt Enhancement.
        Expanded to 50+ acoustic keywords/descriptors covering:
        - Musical instruments (guitar, piano, drums, etc.)
        - Genres (techno, jazz, classical, etc.)
        - Environmental sounds (rain, thunder, wind, etc.)
        - Speech characteristics (narration, broadcast, etc.)
        - Production quality descriptors
        """
        self.knowledge_base = {
            # === MUSICAL INSTRUMENTS ===
            "guitar": "acoustic guitar, crisp resonance, rich harmonics, nylon strings, warm tones, fingerpicking style",
            "piano": "grand piano, sustained pedal, classical reverb, Steinway quality, melodic sustain",
            "violin": "string instruments, bow techniques, classical articulation, rich overtones, concert hall acoustics",
            "drums": "drum kit, punchy kick, crisp snare, hi-hat patterns, tight tuning, professional mixing",
            "bass": "bass guitar, low-end frequencies, sub-bass, warm groove, fingerstyle technique",
            "saxophone": "saxophone, smooth jazz timbre, breathy tone, expressive vibrato, rich harmonics",
            "trumpet": "brass section, bright attack, bold tone, fanfare style, orchestral brass",
            "flute": "woodwind family, breathy tones, ethereal melody, silver flute, classical phrasing",
            "cello": "cello, deep resonant tones, bowing techniques, intimate setting, warm cello timbre",
            "organ": "pipe organ, church acoustics, reverb, sustained chords, Hammond tone",
            "harp": "harp, delicate arpeggios, classical elegance, angelic tone, orchestral texture",
            "synth": "synthesizer, analog warmth, digital precision, pad textures, electronic sound design",
            "trumpet": "trumpet, bold brass, bright trumpet, jazz improvisation, fanfare melodies",
            
            # === GENRES & STYLES ===
            "techno": "128 bpm, sidechain compression, 808 kick, sub-bass frequencies, electronic dynamic range, four-on-floor beat",
            "jazz": "swing rhythm, improvisation, complex chords, blue notes, jazz trio, cool jazz vibe",
            "classical": "orchestral arrangement, classical forms, symphonic texture, chamber music, baroque style",
            "rock": "electric guitar distortion, powerful drums, rock anthem, stadium rock, heavy riffs",
            "hip hop": "boom bap drums, jazzy piano, 90s hip hop, lo-fi beats, sample-based production",
            "edm": "electronic dance music, drop section, build-up, synthesizer leads, club mix",
            "pop": "catchy hooks, polished production, radio-ready, synth pop, modern pop sound",
            "blues": "12-bar blues, guitar bending, blues scale, emotional expression, blues shuffle",
            "country": "acoustic country, steel guitar, country folk, storytelling lyrics, Nashville sound",
            "metal": "heavy metal, distortion, double bass drumming, metalcore, thrash metal",
            "folk": "folk acoustic, campfire folk, traditional instruments, storytelling folk, organic sound",
            "reggae": "reggae rhythm, offbeat guitar, reggae bass, dub style, island vibes",
            "ambient": "ambient soundscape, atmospheric pads, drone textures, meditative, ethereal ambient",
            "cinematic": "cinematic orchestra, film score, dramatic arrangement, orchestral swelling, movie soundtrack",
            "electronic": "electronic production, synthesized sounds, digital effects, modular synth, computer music",
            "bossa nova": "bossa nova rhythm, nylon guitar, soft percussion, Brazilian jazz, bossa groove",
            "synthwave": "synthwave, 80s analog synth, retro vibes, neon sounds, synthwave nostalgia",
            
            # === SPEECH & VOICE ===
            "speech": "close-mic, voice-over quality, clear diction, noise-free, isolated vocal track, broadcast ready",
            "narrator": "narrative voice, storytelling tone, expressive narration, audiobook style, character voices",
            "news": "news anchor, broadcast quality, authoritative tone, professional delivery, studio recording",
            "podcast": "podcast audio, conversational tone, radio quality, interview setup, balanced levels",
            "whisper": "intimate whisper, soft spoken, quiet atmosphere, close-up recording, ASMR style",
            "female voice": "female vocal, soprano range, warm female tone, natural feminine voice",
            "male voice": "male vocal, baritone range, deep male voice, authoritative male voice",
            "child": "child voice, young speaker, natural child's voice, playful tone, kid speaking",
            "teacher": "educational voice, clear instruction, patient explanation, learning environment",
            "meditation": "calm voice, peaceful tone, guided meditation, relaxing instruction, serene voice",
            "radio": "radio broadcast, radio announcer, vintage radio, AM radio sound, old-time radio",
            "telephone": "telephone quality, phone audio, PSTN sound, retro phone, landline effect",
            "voice over": "voice over, professional narration, commercial voice, corporate narration",
            
            # === ENVIRONMENTAL SFX ===
            "rain": "rain sounds, rainfall, raindrops, ambient rain, water droplets, rainy atmosphere",
            "thunder": "thunderstorm, rolling thunder, lightning crack, storm ambience, thunder rumble",
            "wind": "wind sounds, wind blow, gentle breeze, howling wind, air movement",
            "ocean": "ocean waves, sea waves, coastal waves, surf sounds, maritime ambience",
            "forest": "forest ambience, nature sounds, woodland setting, forest ecosystem, peaceful nature",
            "bird": "outdoor field recording, bird songs, avian sounds, chirping birds, natural birdcall",
            "river": "river stream, flowing water, creek sounds, babbling brook, freshwater ambience",
            "fire": "fire crackle, fireplace, burning embers, campfire, crackling fire",
            "thunder": "thunderstorm, electrical storm, rain and thunder, storm weather, dramatic thunder",
            "city": "urban sounds, city ambience, traffic noise, metropolitan sounds, street sounds",
            "restaurant": "ambient field recording, restaurant chatter, clinking glasses, dining ambience",
            "crowd": "crowd sounds, people talking, ambient crowd, large group, public space",
            "car": "car engine, vehicle sounds, automotive audio, driving ambience, car interior",
            "train": "train sounds, railway, train tracks, train whistle, rail travel",
            "airplane": "airplane sounds, jet engine, aircraft, aviation ambience, airport sounds",
            "footsteps": "footsteps, walking sounds, footsteps on surface, footstep effects, walking in various environments",
            "door": "door sounds, door opening, door closing, door slam, wooden door, creaking door",
            "glass": "glass breaking, shatter effect, glass crash, brittle glass, breaking sounds",
            "water": "water sounds, splashing, liquid sounds, water droplet, flowing water",
            "clock": "clock ticking, timepiece, mechanical clock, pendulum clock, tick tock",
            "dog": "dog bark, canine sounds, dog vocalization, puppy sounds, pet sounds",
            "cat": "cat meow, cat sounds, feline vocalization, purring cat, kitten meow",
            "horse": "horse sounds, galloping, horse whinny, equestrian sounds, hooves",
            "helicopter": "helicopter rotor, helicopter sounds, rotary aircraft, chopper sounds",
            "sword": "sword clash, metal weapon, sword fighting, steel sounds, weapon effects",
            "laser": "laser sound, sci-fi laser, beam weapon, futuristic sound, pew pew",
            "explosion": "explosion sound, blast effect, bomb explosion, impact sound, dramatic boom",
            
            # === PRODUCTION QUALITY ===
            "studio": "studio recording, professional studio, high fidelity, clean recording, studio quality",
            "live": "live performance, concert recording, live sound, audience ambience, live venue",
            "ambient": "ambient recording, environmental audio, spatial audio, immersive sound, atmospheric",
            "mono": "mono audio, single channel, mono recording, mono mixdown",
            "stereo": "stereo sound, stereo recording, wide stereo, stereo field, spatial audio",
            "high quality": "high fidelity, lossless audio, hi-res audio, premium quality, professional recording",
            "realistic": "realistic audio, true-to-life, natural sound, authentic reproduction, lifelike",
            "spatial": "spatial audio, 3D audio, immersive audio, Dolby Atmos, surround sound",
            "binaural": "binaural recording, 3D audio, headphone optimized, spatial realism",
            "clean": "clean audio, noise-free, pristine recording, no artifacts, clean mix",
            "crisp": "crisp audio, clear highs, defined transients, sharp sound, clean reproduction",
            "warm": "warm audio, warm tones, analog warmth, vintage sound, rich low-end",
            "bright": "bright audio, bright tone, clear highs, airy sound, treble presence",
            "balanced": "balanced mix, well-balanced, even frequency response, professional mix",
        }

    def fetch_retrieved_context(self, prompt: str) -> str:
        """
        Simulates retrieving chunks from a vector database based on prompt keywords.
        Uses keyword matching to find relevant acoustic descriptors.
        """
        prompt_lower = prompt.lower()
        retrieved_pieces = []
        
        for key, context in self.knowledge_base.items():
            if key in prompt_lower:
                if context not in retrieved_pieces:
                    retrieved_pieces.append(context)
        
        if retrieved_pieces:
            return ", ".join(retrieved_pieces)
        
        # Default fallback context for overall audio enhancement
        return "high quality, clear audio, realistic soundscape, lossless format, professional recording, studio quality"

    def enhance(self, prompt: str) -> str:
        """
        Constructs a new prompt by appending retrieved context, simulating a RAG pattern.
        """
        retrieved_context = self.fetch_retrieved_context(prompt)
        enhanced_prompt = f"{prompt} | Features: {retrieved_context}"
        return enhanced_prompt
    
    def get_coverage_stats(self, prompts: list) -> dict:
        """Analyze how many prompts match keywords in the knowledge base"""
        matched = 0
        matched_keywords = set()
        
        for prompt in prompts:
            prompt_lower = prompt.lower()
            for key in self.knowledge_base:
                if key in prompt_lower:
                    matched += 1
                    matched_keywords.add(key)
                    break
        
        return {
            "matched_count": matched,
            "total_count": len(prompts),
            "coverage_pct": round(100 * matched / len(prompts), 1) if prompts else 0,
            "unique_keywords_matched": len(matched_keywords)
        }


if __name__ == "__main__":
    enhancer = PromptEnhancer()
    
    # Test prompts
    test_prompts = [
        "A cheerful pop song with a fast tempo and bright piano.",
        "Heavy rain falling on a metal roof",
        "A news anchor reporting live from the scene",
        "A rock song with electric guitar and drums"
    ]
    
    print("=" * 60)
    print("RAG Knowledge Base Coverage Test")
    print("=" * 60)
    
    for prompt in test_prompts:
        print(f"\nOriginal: {prompt}")
        print(f"Enhanced: {enhancer.enhance(prompt)}")
        print("-" * 40)
    
    # Coverage stats
    all_prompts = [
        "A cinematic orchestral soundtrack with powerful strings",
        "A jazz piano performance in a cozy lounge",
        "Techno beat with heavy bass and rhythmic percussion",
        "A human female voice speaking clearly",
        "Heavy rain and thunderstorm",
        "Birds chirping in a forest",
        "Restaurant ambience with chatter"
    ]
    
    stats = enhancer.get_coverage_stats(all_prompts)
    print(f"\nKnowledge Base Coverage: {stats['coverage_pct']}% ({stats['matched_count']}/{stats['total_count']})")
    print(f"Unique keywords matched: {stats['unique_keywords_matched']}")