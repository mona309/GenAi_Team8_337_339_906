# 🎤 Presentation Script — Audio/Music Generation Using Diffusion Models

**Team 8 | Meghana Bisa · Mitha M K · Monisha Sharma**

---

## Slide 1 — Title Slide

> *"Good morning/evening. We are Team 8 and our project is about generating audio — music, speech, and sound effects — from text prompts using AI diffusion models. What makes our work unique is not just the generation, but how we measure whether the AI actually understood what you asked for."*

---

## Slide 2 — Abstract / Problem Statement

### What to say:

> *"The core problem we're solving: AI can generate audio that sounds great — but does it sound like what you asked for?*
>
> *Existing evaluation metrics like FAD measure signal quality — basically, does it sound real? But they completely ignore the prompt. A model could generate a perfect-sounding violin piece when you asked for rain sounds — and still score well on traditional metrics.*
>
> *We call this the **evaluation gap**. Our project bridges it."*

### Key phrases to hit:
- "Signal-level metrics ignore intent"
- "We shift from reconstruction accuracy to intent-driven evaluation"
- "GIAF is our answer to the evaluation gap"

---

## Slide 3 — Use Cases

### What to say:

> *"Where does this matter in the real world?*
>
> *First — **Automated Music Production.** A film composer can type 'emotional piano piece for a sad farewell scene' and get a usable draft in seconds.*
>
> *Second — **Sound Design for Games.** Game developers can generate footsteps on gravel, distant thunder, tavern ambience — on demand, without recording studios.*
>
> *These aren't hypothetical. The tech exists today — the gap is in control and reliability of the output.*"

---

## Slide 4 — Literature Review (9 Papers)

> *"We reviewed 9 key papers. Let me highlight the most relevant ones for our work."*

### Hit these 3 specifically:

**1. Auffusion (2024)**
> *"Auffusion was the first to show that text-to-image diffusion architectures could be adapted for audio. Strong CLAP alignment scores — but limited fine-grained control and high compute cost. We use a similar CLAP-based approach but add our cross-domain evaluation layer."*

**2. Retrieval-Augmented Text-to-Audio (2023)**
> *"This paper used CLAP embeddings to retrieve audio-text pairs and guide generation — directly inspired our RAG module. The key weakness was it still failed on completely unseen sounds. Our lightweight keyword KB sidesteps this by being deterministic."*

**3. MusicEval (2025)**
> *"The first expert-rated dataset for text-to-music. It highlighted that automated metrics alone can't capture musical quality. This motivated our POAS metric — it's automated but prompt-grounded, bridging the gap between objective and subjective evaluation."*

---

## Slide 5 — Novel Framework Contributions

### What to say:

> *"We introduce two original contributions:*
>
> **1. GIAF — Generative Intent Alignment Framework**
> *Instead of asking 'does this audio sound real?', GIAF asks 'does this audio match what the user actually wanted?' We measure this at three levels — per sample, per domain, and globally.*
>
> **2. Unified Cross-Domain Diffusion Framework**
> *We run a single evaluation protocol across speech, music, and sound effects simultaneously. No prior work does cross-domain comparison under identical metrics. This lets us directly compare how well the AI transfers its understanding across very different acoustic worlds."*

---

## Slide 6 — System Architecture (Pipeline)

### Walk through LEFT to RIGHT:

```
[Text Prompt] → [RAG Enhancer] → [CLAP Text Encoder] → [AudioLDM 2 / SpeechT5] → [Audio Output]
                                                                                          ↓
                                                                              [CLAP Audio Encoder]
                                                                                          ↓
                                                                              [GIAF: POAS / CRI / CDTS]
```

### What to say for each stage:

**Text Prompt**
> *"User types a natural language prompt — for example, 'A soft acoustic guitar melody in a quiet room'."*

**RAG Enhancer**
> *"Before the prompt even reaches the AI model, our RAG module looks for keywords. It finds 'guitar' — and automatically appends: 'high quality, crisp acoustic resonance, rich harmonics, 44.1kHz recording'. The AI now gets a much richer, more specific description."*

**CLAP Text Encoder**
> *"CLAP — Contrastive Language-Audio Pretraining — encodes the enriched prompt into a vector embedding. Think of it as converting words into a numerical representation that lives in the same space as audio representations."*

**AudioLDM 2 / SpeechT5**
> *"For music and sound effects, AudioLDM 2 uses latent diffusion — it starts from pure Gaussian noise and gradually denoises it into a mel-spectrogram conditioned on that CLAP embedding. For speech specifically, we use Microsoft SpeechT5, a dedicated TTS model, because diffusion models aren't great at generating speech content."*

**GIAF Evaluation**
> *"After generation, we encode the audio through CLAP's audio encoder. We then compare the text embedding and audio embedding — that cosine similarity IS our POAS score."*

---

## Slide 7 — Technical Concepts (GenAI Units)

### Map to course units — say this clearly:

> *"Our project integrates three GenAI concepts from the syllabus:*
>
> **Unit 1 & 4 — LLMs:** CLAP is trained on 630k audio-text pairs and acts as our multimodal language model, grounding audio generation in text semantics.*
>
> **Unit 2 — Prompt Engineering & RAG:** Our RAG module is a direct application of retrieval-augmented generation. Given a short prompt, we retrieve acoustic descriptors from a knowledge base and augment the prompt. This is the same pattern used in production RAG systems — ours is lightweight but functionally identical.*
>
> **Core Generation — Diffusion Models:** AudioLDM 2 is a latent diffusion model. It learns to reverse a noise-addition process, guided by the text embedding, to produce coherent audio."*

---

## Slide 8 — How Diffusion Works (If Asked)

### Simple 3-step explanation:

> *"Imagine taking a photo and slowly covering it with static noise until it's completely unrecognizable — that's the **forward process**.*
>
> *The model learns to do the reverse — given a noisy latent and a text description, it predicts and removes the noise step by step — **20 steps in our system** — until a clean audio spectrogram emerges.*
>
> *The text embedding steers every denoising step via cross-attention layers, so the final audio reflects the conditioning prompt."*

---

## Slide 9 — RAG Module Deep Dive

### What to say:

> *"RAG — Retrieval-Augmented Generation — is usually associated with chatbots retrieving documents. We apply it to audio generation.*
>
> *Our `PromptEnhancer` class has a local knowledge base of 7 acoustic concepts. When a prompt arrives, we scan it for keywords. If we find 'piano', we append: 'grand piano, sustained pedal, classical reverb, high fidelity audio' to the prompt.*
>
> *This is confirmed in `app.py` line 39 — every single generation request goes through this module.*
>
> *The effect? Prompts with matched keywords score **~20% higher** on POAS than unmatched prompts receiving only the generic fallback descriptor."*

---

## Slide 10 — GIAF Metrics Explained

### POAS

> *"POAS — Prompt-to-Audio Similarity. We encode both the enhanced prompt and the generated audio using CLAP, and compute cosine similarity between the two vectors. Score of 1 = perfect semantic match. Score of 0 = no alignment. It's computed per sample — so we know exactly which prompts the model handled well and which it failed."*

### CRI

> *"CRI — Cross-domain Robustness Index. This measures how consistent the model is **within** each domain. We compute the standard deviation of POAS scores per domain, then average across domains. A low CRI means the model handles all prompts in music (or speech, or SFX) equally well. A high CRI reveals some prompts are easy and some are hard — inconsistent behaviour."*

### CDTS

> *"CDTS — Cross-domain Transfer Score. Simple but powerful. It's the global mean POAS across all 30 samples and all 3 domains. One number to summarize 'how good is this system overall at understanding prompts across different audio types?'"*

---

## Slide 11 — Experimental Results

### Correct numbers to use:

| Domain | Samples | Mean POAS | Std |
|---|---|---|---|
| Music | 10 | **0.4082** | 0.0495 |
| SFX | 10 | **0.3072** | 0.1236 |
| Speech | 10 | **0.2846** | 0.1705 |
| **Global** | **30** | **0.3333** | 0.1360 |

**Global CRI = 0.1360 | Global CDTS = 0.3333**

### What to say:

> *"We ran 30 prompts — 10 per domain — through the full pipeline. Here's what we found:*
>
> *Music achieves the highest POAS of 0.4082. This makes sense — CLAP was largely trained on music-captioning datasets, so it understands musical semantics well.*
>
> *Sound effects score 0.3072. Event-level sounds like keyboards and car engines score high (0.4772, 0.4360). Diffuse atmospheric sounds like rain and thunder score lower — the model generates audio that 'sounds like rain' but the semantic alignment is weaker.*
>
> *Speech scores 0.2846 with the highest within-domain variance (0.1705). This is interesting — some prompts like 'person whispering' scored 0.5431, while 'teacher explaining' scored only 0.0073. This shows SpeechT5 sometimes generates audio that doesn't match the specific scenario described.*
>
> *The global CDTS of 0.3333 tells us that on average, about a third of the maximum possible semantic alignment is achieved — there's clear room for improvement, which we identify in future work."*

---

## Slide 12 — Novelty Summary

### What makes this different from existing work:

> *"Three things distinguish our work:*
>
> **1. Cross-domain evaluation under a single metric.** No prior paper evaluates speech, music, AND SFX under identical CLAP-based alignment metrics simultaneously.*
>
> **2. Intent-first evaluation philosophy.** GIAF flips the evaluation question — from 'does this sound real?' to 'does this match what was asked?'*
>
> **3. RAG for audio conditioning.** Applying retrieval augmentation to the prompt enrichment stage for a latent diffusion audio system — with quantified improvement in POAS — is novel in the student project space."*

---

## Slide 13 — Conclusion & Future Work

### What to say:

> *"To conclude: we built a complete text-to-audio pipeline with generation, RAG-based prompt enrichment, and a novel three-metric evaluation framework.*
>
> *Our key finding: semantic alignment in text-to-audio generation is highly domain-dependent. Music handles prompts well. Speech is inconsistent. SFX has wide variance based on event type.*
>
> *Future work: expand the RAG knowledge base from 7 to 50+ concepts, replace keyword matching with a dense neural retriever, and validate GIAF scores against human listener studies."*

---

## 🛡️ Likely Q&A Questions and Answers

**Q: How is your RAG different from standard RAG?**
> *"Standard RAG uses a vector database and neural retriever. Ours uses keyword matching over a local dictionary. Same architectural pattern, lighter implementation. Advantage: fully offline, deterministic, reproducible — no API costs or internet required."*

**Q: Why is speech POAS so variable?**
> *"SpeechT5 generates phonetically plausible speech but doesn't always faithfully reproduce the content scenario described in the prompt. Since we weight ASR-based semantic matching at 90% for speech, a mismatch between the generated speech content and the prompt text directly tanks the score."*

**Q: Why not use FAD?**
> *"FAD measures distributional distance from real audio — it tells you if the audio 'sounds real'. It completely discards the prompt. We deliberately chose POAS because it keeps the prompt in the evaluation loop, which is the point of our work. FAD is a complementary metric we plan to add in future work."*

**Q: Is CLAP the right metric for speech?**
> *"Great question — actually no, which is why we modified POAS for speech. CLAP was trained on environmental audio and music, not speech content. Pure CLAP scores for speech are unreliable. That's why we blend it 10/90 with ASR-based semantic similarity for the speech domain."*

**Q: How did you validate that RAG improves results?**
> *"From the results CSV: guitar and piano prompts that received matched descriptors scored 0.484–0.485 POAS. Unmatched music prompts scored 0.331–0.365. That's roughly a 20% improvement for keyword-matched prompts."*
