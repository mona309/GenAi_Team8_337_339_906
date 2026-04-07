#!/usr/bin/env python3
"""
Overnight Experiment Script for Text-to-Audio Q1 Journal Evaluation
Runs comprehensive ablation study: RAG vs No-RAG, Multiple Models, Extended Prompts

Usage: python overnight_experiment.py [--skip-generation] [--models audioldm2]
"""

import os
import sys
import argparse
import json
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.evaluation import AudioEvaluator
from src.rag_enhancer import PromptEnhancer

MODELS = {
    "audioldm2": "cvssp/audioldm2",
    "audioldm2-large": "cvssp/audioldm2-large",
    "musicgen-small": "facebook/musicgen-small",
    "musicgen-medium": "facebook/musicgen-medium",
    "musicldm": "segelo/musicldm",
}

DOMAINS = ["music", "speech", "sfx"]

EXPANDED_PROMPTS = {
    "music": [
        "A cinematic orchestral soundtrack with powerful strings and brass",
        "A lo-fi chill hip hop beat with vinyl crackle and soft drums",
        "A jazz piano performance in a cozy lounge with warm tones",
        "An energetic EDM track with deep bass drops and bright synths",
        "A classical violin solo in a concert hall with emotional expression",
        "A soft acoustic guitar melody with natural resonance",
        "A rock song with electric guitar riffs and powerful drums",
        "A cinematic ambient soundtrack with deep textures",
        "A techno beat with heavy bass and rhythmic percussion",
        "A flute melody in a peaceful forest setting",
        "A drum and bass track with fast breaks and heavy bassline",
        "A classical piano sonata in the style of Mozart",
        "A blues guitar solo with expressive bending and vibrato",
        "A reggae track with offbeat rhythm and warm bass",
        "A country folk song with acoustic guitar and harmonica",
        "A heavy metal track with distorted guitars and double bass",
        "A baroque orchestral piece with harpsichord and strings",
        "A synthwave track with retro analog synths and neon sounds",
        "A bossa nova rhythm with nylon guitar and soft percussion",
        "A trancy house track with ethereal pads and driving beat",
        "A string quartet in a modern minimalist style",
        "A solo cello piece with deep resonant tones",
        "A marching band with brass section and drum cadence",
        "A japanese koto and shakuhachi traditional piece",
        "A african drumming circle with tribal rhythms",
        "A pop ballad with emotional vocals and piano accompaniment",
        "A hip hop track with boom bap drums and jazzy piano",
        "A ambient soundscape with nature sounds and soft pads",
        "A latin salsa track with percussion and brass ensemble",
        "A 1980s new wave synthpop track with catchy melody",
    ],
    "speech": [
        "A high-quality studio recording of a human female voice speaking clearly",
        "A human male voice delivering a speech with clear pronunciation",
        "A professional news anchor speaking with authoritative tone",
        "A child speaking and laughing naturally in a playground",
        "A teacher explaining a complex topic with patient clarity",
        "A person whispering softly in a quiet library setting",
        "A podcast conversation between two hosts with balanced audio",
        "A radio announcer with deep resonant voice",
        "A person speaking on a telephone with slight compression",
        "A storyteller narrating an exciting adventure with expression",
        "A medical professional explaining a procedure calmly",
        "A sports commentator describing game action energetically",
        "A motivational speaker inspiring an audience with passion",
        "A voice over for a nature documentary with calm authority",
        "A historical narrator describing ancient events dramatically",
        "A tutorial voice explaining technical concepts step by step",
        "A meditation guide with peaceful calming instructions",
        "An audiobook narrator reading fiction with character voices",
        "A political figure giving a formal address",
        "A customer service representative speaking politely",
        "An alarm clock voice giving urgent wake up message",
        "A gps navigation voice giving clear directions",
        "An ai assistant responding in a friendly manner",
        "A doctor giving a patient a diagnosis with empathy",
        "A judge delivering a courtroom ruling formally",
    ],
    "sfx": [
        "A highly realistic recording of heavy rain falling on a metal roof",
        "Powerful ocean waves crashing against large coastal rocks",
        "A car engine starting and accelerating smoothly",
        "Fire crackling in a quiet forest with popping embers",
        "Footsteps walking on dry autumn leaves",
        "Birds chirping in a peaceful morning forest",
        "A crowded restaurant with chatter and clinking glasses",
        "A thunderstorm with rolling thunder and heavy rain",
        "A airplane taking off from an airport runway",
        "A train passing by with horn and rail sounds",
        "A dog barking excited to see its owner",
        "A cat meowing softly while stretching",
        "A horse galloping across a field",
        "A helicopter flying overhead and landing",
        "A door closing with a heavy thud",
        "A glass breaking with sharp crash",
        "A water faucet dripping in a silent bathroom",
        "A keyboard typing rapidly on a mechanical keyboard",
        "A printer printing documents continuously",
        "A phone ringing loudly in a quiet room",
        "An elevator ding and doors sliding open",
        "A subway train arriving at a platform",
        "A police siren wailing in the distance",
        "A baby crying with earnest wails",
        "A zombie groaning with guttural sounds",
        "A spaceship taking off with futuristic thrusters",
        "A laser blast with sci-fi impact sound",
        "A sword clashing against another sword",
        "A dragon breathing fire with intense roar",
        "A clock ticking in an antique grandfather clock",
    ]
}

def setup_experiment_dirs(base_dir="outputs/experiments"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"exp_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def load_generator(model_name):
    """Lazy load generator only when needed"""
    from src.generation import AudioGenerator
    model_id = MODELS.get(model_name, model_name)
    print(f"Loading model: {model_id}")
    return AudioGenerator(model_id=model_id, device="cuda")


def load_musicgen(model_name):
    """Load MusicGen model for music generation"""
    import torchaudio
    from transformers import MusicgenForConditionalGeneration, MusicgenProcessor
    
    model_id = MODELS.get(model_name, model_name)
    print(f"Loading MusicGen model: {model_id}")
    
    processor = MusicgenProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id)
    model = model.to("cuda")
    
    class MusicGenWrapper:
        def __init__(self, model, processor):
            self.model = model
            self.processor = processor
            
        def generate(self, prompt: str, output_path: str, **kwargs):
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            audio_length = kwargs.get("audio_length_in_s", 10.0)
            num_inference_steps = kwargs.get("num_inference_steps", 50)
            
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=int(audio_length * 256)
            )
            
            audio = audio_values[0].cpu().numpy()
            
            sample_rate = 32000
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, torch.tensor(audio).unsqueeze(0), sample_rate)
            
            print(f"Saved: {output_path}")
            return output_path
    
    return MusicGenWrapper(model, processor)


def load_model(model_name):
    """Route to appropriate model loader"""
    if "musicgen" in model_name:
        return load_musicgen(model_name)
    else:
        return load_generator(model_name)

def generate_audio_batch(prompts, generator, domain, output_dir, use_rag=True, 
                        num_inference_steps=50, audio_length=None, model_name=None):
    """Generate audio for a batch of prompts"""
    rag = PromptEnhancer() if use_rag else None
    results = []
    
    # MusicGen only generates music - skip speech/sfx
    if "musicgen" in str(type(generator).__name__.lower()) or (model_name and "musicgen" in model_name):
        if domain != "music":
            print(f"  Skipping {domain} for MusicGen (music-only model)")
            return results
    
    for idx, prompt in enumerate(tqdm(prompts, desc=f"Generating {domain} (RAG={use_rag})")):
        if use_rag and rag:
            enhanced = rag.enhance(prompt)
        else:
            enhanced = f"{prompt}. High quality, realistic, clear audio."
        
        # Domain-specific audio length
        if audio_length is None:
            if domain == "sfx":
                audio_len = 7.0
            elif domain == "speech":
                audio_len = 4.0
            else:  # music
                audio_len = 10.0
        else:
            audio_len = audio_length
        
        output_path = output_dir / f"{domain}_{idx}_rag{use_rag}.wav"
        
        # Use model-specific generate method
        try:
            generator.generate(
                prompt=enhanced,
                output_path=str(output_path),
                num_inference_steps=num_inference_steps,
                audio_length_in_s=audio_len
            )
        except Exception as e:
            print(f"  Generation failed for {output_path}: {e}")
            continue
        
        results.append({
            "prompt": prompt,
            "enhanced_prompt": enhanced,
            "domain": domain,
            "use_rag": use_rag,
            "audio_path": str(output_path)
        })
    
    return results

def run_evaluation(audio_results, evaluator, exp_dir, model_name=None):
    """Evaluate all generated audio with all metrics"""
    all_scores = []
    audio_paths = []
    
    musicgen_models = ["musicgen-small", "musicgen-medium"]
    
    for item in tqdm(audio_results, desc="Evaluating audio"):
        # Skip non-music for MusicGen models
        if model_name in musicgen_models and item["domain"] != "music":
            continue
            
        score = evaluator.evaluate_clap(item["enhanced_prompt"], item["audio_path"])
        all_scores.append({
            **item,
            "poas_score": score
        })
        audio_paths.append(item["audio_path"])
    
    df = pd.DataFrame(all_scores)
    
    # Compute GIAF metrics
    cri = float(df.groupby("domain")["poas_score"].std().mean()) if len(df) > 0 else 0
    cdts = float(df["poas_score"].mean()) if len(df) > 0 else 0
    
    domain_stats = df.groupby("domain")["poas_score"].agg(["mean", "std", "count"])
    
    # Compute FAD if we have audio paths and evaluator supports it
    fad_score = None
    if audio_paths and hasattr(evaluator, 'fad_model') and evaluator.fad_model:
        try:
            fad_score = evaluator.evaluate_fad(audio_paths)
        except Exception as e:
            print(f"FAD computation failed: {e}")
    
    return {
        "all_results": all_scores,
        "dataframe": df,
        "cri": cri,
        "cdts": cdts,
        "domain_stats": domain_stats,
        "fad": fad_score
    }

def compare_rag_conditions(df):
    """Compare RAG vs No-RAG results"""
    rag_scores = df[df["use_rag"] == True]["poas_score"]
    no_rag_scores = df[df["use_rag"] == False]["poas_score"]
    
    return {
        "rag_mean": rag_scores.mean(),
        "no_rag_mean": no_rag_scores.mean(),
        "rag_std": rag_scores.std(),
        "no_rag_std": no_rag_scores.std(),
        "improvement": rag_scores.mean() - no_rag_scores.mean()
    }

def generate_report(metrics_dict, exp_dir):
    """Generate final experiment report"""
    report = []
    report.append("=" * 80)
    report.append("OVERNIGHT EXPERIMENT REPORT - Q1 Journal Evaluation")
    report.append("=" * 80)
    report.append(f"Timestamp: {datetime.now().isoformat()}")
    report.append("")
    
    for model_name, model_metrics in metrics_dict.items():
        report.append(f"\n### Model: {model_name}")
        report.append("-" * 40)
        
        for condition, cond_metrics in model_metrics.items():
            report.append(f"\n  Condition: {condition}")
            report.append(f"    CRI: {cond_metrics.get('cri', 'N/A'):.4f}")
            report.append(f"    CDTS: {cond_metrics.get('cdts', 'N/A'):.4f}")
            
            if "rag_comparison" in cond_metrics:
                rc = cond_metrics["rag_comparison"]
                report.append(f"    RAG vs No-RAG improvement: {rc['improvement']:.4f}")
                report.append(f"    RAG mean: {rc['rag_mean']:.4f}")
                report.append(f"    No-RAG mean: {rc['no_rag_mean']:.4f}")
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save to file
    with open(exp_dir / "experiment_report.txt", "w") as f:
        f.write(report_text)
    
    return report_text

def main():
    parser = argparse.ArgumentParser(description="Overnight Experiment Runner")
    parser.add_argument("--skip-generation", action="store_true", 
                        help="Skip audio generation, use existing files")
    parser.add_argument("--models", nargs="+", default=["audioldm2"],
                        choices=list(MODELS.keys()) + ["all"],
                        help="Models to evaluate")
    parser.add_argument("--inference-steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--sample-limit", type=int, default=None,
                        help="Limit samples per domain for quick test")
    args = parser.parse_args()
    
    if "all" in args.models:
        models_to_run = list(MODELS.keys())
    else:
        models_to_run = args.models
    
    exp_dir = setup_experiment_dirs()
    print(f"Experiment directory: {exp_dir}")
    
    evaluator = AudioEvaluator(use_cuda=True)
    
    metrics_dict = {}
    
    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"STARTING MODEL: {model_name}")
        print(f"{'='*60}")
        
        model_metrics = {}
        
        # Load generator
        generator = load_model(model_name)
        
        for use_rag in [True, False]:
            condition = "RAG" if use_rag else "No-RAG"
            print(f"\n--- Running condition: {condition} ---")
            
            # Prepare prompts
            all_prompts = []
            for domain in DOMAINS:
                domain_prompts = EXPANDED_PROMPTS[domain].copy()
                if args.sample_limit:
                    domain_prompts = domain_prompts[:args.sample_limit]
                all_prompts.extend([(p, domain) for p in domain_prompts])
            
            # Generate
            output_dir = exp_dir / model_name / condition.lower().replace("-", "_")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if not args.skip_generation:
                # Group by domain for batch processing
                for domain in DOMAINS:
                    domain_prompts = [p for p, d in all_prompts if d == domain]
                    generate_audio_batch(
                        domain_prompts, generator, domain, output_dir,
                        use_rag=use_rag,
                        num_inference_steps=args.inference_steps,
                        model_name=model_name
                    )
            
            # Evaluate
            audio_results = []
            for domain in DOMAINS:
                domain_prompts = [p for p, d in all_prompts if d == domain]
                for idx, prompt in enumerate(domain_prompts):
                    audio_path = output_dir / f"{domain}_{idx}_rag{use_rag}.wav"
                    if audio_path.exists():
                        rag_enh = PromptEnhancer()
                        enhanced = rag_enh.enhance(prompt) if use_rag else f"{prompt}. High quality, realistic, clear audio."
                        audio_results.append({
                            "prompt": prompt,
                            "enhanced_prompt": enhanced,
                            "domain": domain,
                            "use_rag": use_rag,
                            "audio_path": str(audio_path)
                        })
            
            eval_results = run_evaluation(audio_results, evaluator, exp_dir, model_name=model_name)
            
            # Compute rag comparison if we have both conditions
            model_metrics[condition] = {
                "cri": eval_results["cri"],
                "cdts": eval_results["cdts"],
                "fad": eval_results.get("fad"),
                "domain_stats": eval_results["domain_stats"].to_dict() if hasattr(eval_results["domain_stats"], 'to_dict') else dict(eval_results["domain_stats"]),
                "sample_count": len(eval_results["all_results"])
            }

            # Save intermediate results
            eval_results["dataframe"].to_csv(exp_dir / f"{model_name}_{condition}_results.csv", index=False)

            # Save metrics
            metrics_dict[model_name] = model_metrics

            # Save progress checkpoint
            with open(exp_dir / "checkpoint.json", "w") as f:
                json.dump(metrics_dict, f, indent=2, default=str)
            
            print(f"  Completed {condition}: CRI={eval_results['cri']:.4f}, CDTS={eval_results['cdts']:.4f}" + 
                  (f", FAD={eval_results['fad']:.2f}" if eval_results.get("fad") else ""))
        
        # Compare RAG vs No-RAG for this model
        if "RAG" in model_metrics and "No-RAG" in model_metrics:
            rag_df = pd.read_csv(exp_dir / f"{model_name}_RAG_results.csv")
            no_rag_df = pd.read_csv(exp_dir / f"{model_name}_No-RAG_results.csv")
            
            comparison = compare_rag_conditions(pd.concat([rag_df, no_rag_df]))
            model_metrics["rag_comparison"] = comparison
            
            print(f"\n  >>> RAG Improvement: {comparison['improvement']:.4f}")
    
    # Final report
    generate_report(metrics_dict, exp_dir)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    try:
        from src.visualization import ResultsVisualizer
        viz = ResultsVisualizer(str(exp_dir))
        all_results = {}
        for csv_file in exp_dir.glob("*_results.csv"):
            name = csv_file.stem
            all_results[name] = pd.read_csv(csv_file)
        
        if all_results:
            viz.plot_model_comparison_bar(viz.create_comparison_table(all_results))
            viz.plot_rag_comparison(all_results)
            viz.plot_domain_heatmap(all_results)
            viz.plot_domain_breakdown(viz.create_comparison_table(all_results))
            viz.create_summary_figure(all_results, metrics_dict)
            viz.generate_latex_table(viz.create_comparison_table(all_results))
            print(f"Figures saved to: {viz.figures_dir}")
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()