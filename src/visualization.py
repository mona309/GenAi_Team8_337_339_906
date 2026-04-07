#!/usr/bin/env python3
"""
Visualization module for Q1 Journal Results
Generates: comparison tables, bar charts, heatmaps, domain analysis plots
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class ResultsVisualizer:
    def __init__(self, results_dir="outputs/experiments"):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
    
    def load_all_results(self):
        """Load all experiment result CSVs"""
        results = {}
        for csv_file in self.results_dir.glob("*_results.csv"):
            name = csv_file.stem
            results[name] = pd.read_csv(csv_file)
        return results
    
    def create_comparison_table(self, all_results):
        """Create formatted comparison table for paper"""
        rows = []
        
        for name, df in all_results.items():
            # Parse model and condition from filename
            parts = name.split("_")
            model = parts[0] if parts else "unknown"
            condition = parts[-1] if len(parts) > 1 else "unknown"
            
            if "poas_score" in df.columns:
                row = {
                    "Model": model,
                    "Condition": condition,
                    "N": len(df),
                    "Mean POAS": df["poas_score"].mean(),
                    "Std POAS": df["poas_score"].std(),
                }
                
                # Add domain-specific scores
                for domain in ["music", "speech", "sfx"]:
                    domain_df = df[df["domain"] == domain]
                    if len(domain_df) > 0:
                        row[f"{domain.title()} POAS"] = domain_df["poas_score"].mean()
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def plot_model_comparison_bar(self, comparison_df, metric="Mean POAS", output_name="model_comparison.png"):
        """Bar chart comparing models"""
        if comparison_df.empty:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group by model
        models = comparison_df["Model"].unique()
        x = np.arange(len(models))
        width = 0.35
        
        if metric in comparison_df.columns:
            values = [comparison_df[comparison_df["Model"]==m][metric].mean() for m in models]
            ax.bar(x, values, width, label=metric, color='steelblue')
        
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Across Models')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / output_name, dpi=150)
        plt.close()
        print(f"Saved: {self.figures_dir / output_name}")
    
    def plot_rag_comparison(self, all_results):
        """Compare RAG vs No-RAG performance"""
        rag_scores = []
        no_rag_scores = []
        
        for name, df in all_results.items():
            if "rag" in name.lower():
                if "RAG" in name or "ragtrue" in name.lower():
                    rag_scores.extend(df["poas_score"].tolist())
                else:
                    no_rag_scores.extend(df["poas_score"].tolist())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Box plot
        data = [rag_scores, no_rag_scores]
        bp = ax1.boxplot(data, labels=["RAG", "No-RAG"], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax1.set_ylabel("POAS Score")
        ax1.set_title("RAG vs No-RAG Distribution")
        
        # Bar comparison
        means = [np.mean(rag_scores), np.mean(no_rag_scores)]
        stds = [np.std(rag_scores), np.std(no_rag_scores)]
        ax2.bar(["RAG", "No-RAG"], means, yerr=stds, capsize=5, 
                color=['steelblue', 'coral'], alpha=0.8)
        ax2.set_ylabel("Mean POAS")
        ax2.set_title("RAG vs No-RAG Mean (+ std)")
        
        improvement = ((means[0] - means[1]) / means[1]) * 100 if means[1] != 0 else 0
        ax2.text(0.5, 0.95, f"Improvement: {improvement:.1f}%", 
                transform=ax2.transAxes, ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "rag_comparison.png", dpi=150)
        plt.close()
        print(f"Saved: {self.figures_dir / 'rag_comparison.png'}")
        
        return {
            "rag_mean": np.mean(rag_scores),
            "no_rag_mean": np.mean(no_rag_scores),
            "improvement_pct": improvement
        }
    
    def plot_domain_heatmap(self, all_results):
        """Create domain × model heatmap of CDTS scores"""
        data = {}
        
        for name, df in all_results.items():
            model = name.split("_")[0]
            condition = name.split("_")[-1]
            
            for domain in ["music", "speech", "sfx"]:
                domain_df = df[df["domain"] == domain]
                if len(domain_df) > 0:
                    key = f"{model}_{condition}"
                    if key not in data:
                        data[key] = {}
                    data[key][domain] = domain_df["poas_score"].mean()
        
        if not data:
            return
        
        df_heatmap = pd.DataFrame(data).T
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=df_heatmap.values.mean(), vmin=0, vmax=1)
        plt.title("Domain × Model POAS Heatmap")
        plt.xlabel("Audio Domain")
        plt.ylabel("Model + Condition")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "domain_heatmap.png", dpi=150)
        plt.close()
        print(f"Saved: {self.figures_dir / 'domain_heatmap.png'}")
    
    def plot_domain_breakdown(self, comparison_df):
        """Plot domain-wise breakdown comparison"""
        domains = ["Music POAS", "Speech POAS", "Sfx POAS"]
        existing_domains = [d for d in domains if d in comparison_df.columns]
        
        if not existing_domains:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(comparison_df))
        width = 0.25
        
        for i, domain in enumerate(existing_domains):
            offset = (i - len(existing_domains)/2 + 0.5) * width
            bars = ax.bar(x + offset, comparison_df[domain], width, label=domain.replace(" POAS", ""))
        
        ax.set_xlabel("Model")
        ax.set_ylabel("POAS Score")
        ax.set_title("Domain-wise POAS Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df["Model"], rotation=45, ha='right')
        ax.legend(title="Domain")
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "domain_breakdown.png", dpi=150)
        plt.close()
        print(f"Saved: {self.figures_dir / 'domain_breakdown.png'}")
    
    def generate_latex_table(self, comparison_df, output_name="results_table.tex"):
        """Generate LaTeX table for paper"""
        if comparison_df.empty:
            return
        
        # Format numbers
        latex = "\\begin{table}[t]\n\\centering\n"
        latex += "\\caption{Model Comparison Results}\n"
        latex += "\\label{tab:results}\n"
        latex += "\\begin{tabular}{|l|c|c|c|c|}\n\\hline\n"
        latex += "Model & Condition & N & Mean POAS $\\uparrow$ & Std \\\\ \\hline\n"
        
        for _, row in comparison_df.iterrows():
            model = row.get("Model", "")
            cond = row.get("Condition", "")
            n = row.get("N", 0)
            mean = row.get("Mean POAS", 0)
            std = row.get("Std POAS", 0)
            latex += f"{model} & {cond} & {n} & {mean:.4f} & {std:.4f}\\\\ \n"
        
        latex += "\\hline\n\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        with open(self.figures_dir / output_name, 'w') as f:
            f.write(latex)
        
        print(f"Saved: {self.figures_dir / output_name}")
        return latex
    
    def create_summary_figure(self, all_results, metrics_dict):
        """Create comprehensive summary figure with multiple subplots"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Overall comparison
        ax1 = fig.add_subplot(2, 2, 1)
        model_means = {}
        for name, df in all_results.items():
            model = name.split("_")[0]
            if model not in model_means:
                model_means[model] = []
            model_means[model].append(df["poas_score"].mean())
        
        model_labels = list(model_means.keys())
        model_vals = [np.mean(v) for v in model_means.values()]
        ax1.barh(model_labels, model_vals, color='steelblue')
        ax1.set_xlabel("Mean POAS")
        ax1.set_title("Overall Model Performance")
        ax1.set_xlim(0, 1)
        
        # 2. Domain comparison
        ax2 = fig.add_subplot(2, 2, 2)
        domain_means = {"music": [], "speech": [], "sfx": []}
        for name, df in all_results.items():
            for domain in domain_means:
                domain_df = df[df["domain"] == domain]
                if len(domain_df) > 0:
                    domain_means[domain].append(domain_df["poas_score"].mean())
        
        domain_labels = list(domain_means.keys())
        domain_vals = [np.mean(v) if v else 0 for v in domain_means.values()]
        ax2.bar(domain_labels, domain_vals, color=['#2ecc71', '#3498db', '#e74c3c'])
        ax2.set_ylabel("Mean POAS")
        ax2.set_title("Cross-Domain Performance")
        ax2.set_ylim(0, 1)
        
        # 3. RAG impact
        ax3 = fig.add_subplot(2, 2, 3)
        rag_impact = self.plot_rag_comparison(all_results)
        ax3.axis('off')
        ax3.text(0.5, 0.5, f"RAG Improvement: {rag_impact['improvement_pct']:.1f}%\n"
                  f"RAG Mean: {rag_impact['rag_mean']:.4f}\n"
                  f"No-RAG Mean: {rag_impact['no_rag_mean']:.4f}",
                  ha='center', va='center', fontsize=14,
                  bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        # 4. Sample count
        ax4 = fig.add_subplot(2, 2, 4)
        sample_counts = {name: len(df) for name, df in all_results.items()}
        ax4.barh(list(sample_counts.keys()), list(sample_counts.values()), color='lightgray')
        ax4.set_xlabel("Number of Samples")
        ax4.set_title("Samples per Configuration")
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "summary_figure.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.figures_dir / 'summary_figure.png'}")


def generate_all_visualizations(results_dir="outputs/experiments"):
    """Main function to generate all visualizations"""
    viz = ResultsVisualizer(results_dir)
    
    print("Loading results...")
    all_results = viz.load_all_results()
    
    if not all_results:
        print("No result files found!")
        return
    
    print("Generating visualizations...")
    
    # Create comparison table
    comparison_df = viz.create_comparison_table(all_results)
    print("\nComparison Table:")
    print(comparison_df)
    
    # Generate plots
    viz.plot_model_comparison_bar(comparison_df)
    viz.plot_rag_comparison(all_results)
    viz.plot_domain_heatmap(all_results)
    viz.plot_domain_breakdown(comparison_df)
    viz.generate_latex_table(comparison_df)
    viz.create_summary_figure(all_results, {})
    
    # Save comparison table
    comparison_df.to_csv(viz.figures_dir / "comparison_table.csv", index=False)
    
    print(f"\nAll figures saved to: {viz.figures_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="outputs/experiments")
    args = parser.parse_args()
    generate_all_visualizations(args.results_dir)