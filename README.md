# Text-to-Audio GenAI Mini Project

**Team Members:**
- Meghana Bisa (PES2UG23CS337)
- Mitha M K (PES2UG23CS339)
- Monisha Sharma (PES2UG23CS906)
**Semester:** 6th Semester, PES University

---

### Slide 1: Title Slide
**Project Title:** Modular Pipeline for Text-to-Audio Generation using Diffusion Models
**Domain:** Generative AI System (LLMs, Diffusion Models)
**Team:** 8

---

### Slide 2: Problem Statement and Abstract
**Abstract:**
Generating high-fidelity audio from free-form text remains highly challenging due to text-audio alignment, semantic richness, and temporal coherence. Our project builds a modular GenAI pipeline capable of transforming short, natural language instructions into high-quality soundscapes, music, and speech.
**Problem Statement:**
Standard text-to-audio models struggle with brevity (short prompts lacking acoustic detail). We tackle this by implementing a modular text-to-audio pipeline enhanced with an LLM-inspired RAG approach to automatically enrich prompt semantics, executing it through a Latent Diffusion Model (AudioLDM 2) conditioned by Flan-T5.

---

### Slide 3: Use Case of the Project
1. **Foley & Sound Design Automation**: Game developers and film editors can synthesize missing environmental sounds (SFX) instantly.
2. **Dynamic Background Music Generation**: Content creators can generate royalty-free background music specifying tempo, genre, and mood.
3. **Accessibility**: Generating realistic and dynamic audio descriptions for visually impaired digital experiences.

---

### Slide 4: Novelty of the Proposed Work
- **Multi-modal Conditioning**: Combines Large Language Models (Flan-T5) with Latent Diffusion to bridge textual semantics and audio latents.
- **RAG-based Prompt Enhancement (Unit-2)**: Traditional approaches expect prompt engineering directly from the user. We augment user prompts with an offline Retrieval-Augmented Generation (RAG) feature to inject rich acoustic details based on matching keywords.
- **Comprehensive Evaluation Suite**: Moves beyond subjective hearing tests by implementing robust mathematical evaluation frameworks including CLAP alignment, FAD distance, and Cross-domain tracking metrics (CRI, CDTS).

---

### Slide 5: Validation Metrics (Proposed)
Our pipeline uses rigorous, state-of-the-art Generative AI metrics:
1. **CLAP Similarity Score**: Measures how well the generated audio aligns with the original text prompt in a joint embedding space.
2. **FAD (Fréchet Audio Distance)**: Compares VGGish embeddings of synthetic audio against a set of real-world "reference" audio files to determine overall audio realism / fidelity.
3. **POAS (Prompt-to-Audio Similarity)**: Our custom wrapper over CLAP measuring verbatim semantic intent alignment.
4. **CRI (Cross-domain Robustness Index)**: Measures standard deviation of POAS scores across distinct audio domains (Speech, Music, SFX).
5. **CDTS (Cross-domain Transfer Score)**: Measures cross-domain generalization capability.

---

### Slide 6: Existing Work / Literature Review
1. **AudioLDM:** Tang, H., et al. (2023). *AudioLDM: Text-to-Audio Generation with Latent Diffusion Models*. Proposed the foundational Latent space modeling for continuous audio.
2. **Flan-T5 (LLM Instruction Tuning):** Chung, H. W., et al. (2022). *Scaling Instruction-Finetuned Language Models*. We use Flan-T5 as the core text encoder to handle complex instructional prompts for diffusion conditioning.
3. **RAG (Retrieval-Augmented Generation):** Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. Our project leverages this principle (Unit-2) to fetch acoustic descriptive features prior to generation.

---

### Slide 7: Technical Aspects of GenAI Concepts Used
This project intrinsically integrates exactly the units required for the GenAI Project Scope:
- **Unit 1 & 4 (LLMs):** Uses **Flan-T5** (a large language model) directly within the pipeline framework to deeply encode complex textual prompts.
- **Unit 2 (Prompt Engineering & RAG):** Implements a `PromptEnhancer` class that intercepts user prompts and performs pseudo-retrieval (RAG) to append rich acoustic characteristics (e.g. converting "A piano" -> "A piano, grand piano, sustained pedal, classical reverb, high fidelity audio").
- **Core Generation Mechanism:** Uses **AudioLDM 2**, a state-of-the-art text-to-audio Latent Diffusion model to generate the actual new content.

### Implementation Stack:
- `diffusers`, `transformers` (Hugging Face)
- `laion-clap` (Text-Audio Feature Encoding)
- `frechet_audio_distance` (VGGish realism metric)
- `PyTorch` (GPU Tensor Computation)

---

### Slide 8: Validation Metrics (Implemented / Showing Code)
We have implemented a custom `src/evaluation.py` module executing the metrics over a test batch `data/prompts.csv`. 
To run the full pipeline and generate the final results table `outputs/results_table.csv`, run:
```bash
pip install -r requirements.txt
python main.py
```
> Outputs include: generated `.wav` files and a metric console dump aggregating CRI and CDTS.

---
*Draft for internal review formatting prior to PPT creation.*