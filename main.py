import os

# Crucial fixes for TensorFlow/PyTorch ABI and Protobuf Segfaults in Linux:
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python' # Prevents C++ descriptor pool crashes
os.environ['USE_TF'] = '0' # Stops HuggingFace Transformers from implicitly loading TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppresses noisy TF warnings

import pandas as pd
from tqdm import tqdm

from src.generation import AudioGenerator
from src.evaluation import AudioEvaluator
from src.rag_enhancer import PromptEnhancer

def main():
    print("--- GenAI Text-to-Audio Mini Project ---")
    
    # Setup Output Directories
    os.makedirs("outputs/audio", exist_ok=True)
    os.makedirs("data/reference_audio", exist_ok=True) # Placeholder for FAD references
    
    dataset_path = "data/prompts.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset {dataset_path} not found.")
        
    df = pd.read_csv(dataset_path)
    
    # Limit to 5 samples for brief execution if needed. 
    # For full run, comment out the limit.
    #df = df.head(5)
    df = pd.concat([
        df[df["domain"] == "music"].head(10),
        df[df["domain"] == "speech"].head(10),
        df[df["domain"] == "sfx"].head(10),
    ], ignore_index=True)

    generator = AudioGenerator()
    evaluator = AudioEvaluator()
    rag = PromptEnhancer()

    results = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Pipeline"):
        original_prompt = row['prompt']
        domain = row['domain']
        
        # 1. Option: RAG Prompt Enhancement
        enhanced_prompt = rag.enhance(original_prompt)
        
        # 2. Generate Audio
        audio_filename = f"outputs/audio/{domain}_{index}.wav"
        generator.generate(
            prompt=enhanced_prompt, 
            output_path=audio_filename, 
            num_inference_steps=40, # reduced for speed
            audio_length_in_s=5.0
        )
        
        # 3. Evaluate alignment (CLAP/POAS)
        poas_score = evaluator.evaluate_clap(enhanced_prompt, audio_filename)
        
        results.append({
            "id": index,
            "domain": domain,
            "original_prompt": original_prompt,
            "enhanced_prompt": enhanced_prompt,
            "poas_score": poas_score, # Prompt-to-Audio Similarity
            "audio_file": audio_filename
        })

    # Save initial results
    results_df = pd.DataFrame(results)
    output_csv = "outputs/results_table.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved raw generation results to {output_csv}")

    transfer_log_path = "outputs/domain_transfer_results.csv"
    transfer_df = evaluator.save_domain_transfer_log(results, transfer_log_path)
    print(f"Saved domain transfer log to {transfer_log_path}")
    
    # 4. Global Metrics: FAD, CRI, CDTS
    agg_metrics = evaluator.compute_aggregate_metrics(results)

    print("\n--- Final Project Evaluation ---")
    print(f"1. Average POAS (CLAP): {results_df['poas_score'].mean():.4f}")
    print(f"2. CRI (Cross-domain Robustness): {agg_metrics['CRI_Robustness_Variance']}")
    print(f"3. CDTS (Cross-domain Transfer): {agg_metrics['CDTS_Transfer_Mean']}")
    print("4. Domain Transfer by class:")
    print(transfer_df.to_string(index=False))

    print("\nPipeline execution complete! Check outputs/ directory for generated logs and audio.")

if __name__ == "__main__":
    main()
