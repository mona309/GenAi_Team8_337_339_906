"""
batch_eval.py  (v2 — improved)
================================
Key changes vs v1
-----------------
1. Pass original_prompt as asr_reference to evaluate_poas()
   When RAG is enabled, the enhanced prompt (text_prompt) can be 60-120
   words long. Using that as the WER reference inflates the word error rate
   because Whisper only transcribes what was actually *spoken* — the short
   original TTS text. We now pass the original prompt separately so the
   semantic score reflects true speech intelligibility.

2. Transcript logged in results CSV
   The transcript column from evaluate_poas() is now stored in the output
   CSV so you can inspect what Whisper heard for each speech sample.

3. Everything else (plot, summary, CLI) unchanged.

Usage
-----
    python batch_eval.py                          # uses prompts.csv in CWD
    python batch_eval.py --input my_prompts.csv
    python batch_eval.py --input prompts.csv --output-dir results/
    python batch_eval.py --skip-rag-off           # only run RAG-on pass
"""

import argparse
import os
import sys
import uuid
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.generation  import AudioGenerator
from src.evaluation  import AudioEvaluator
from src.rag_enhancer import PromptEnhancer

# ── Colour palette ────────────────────────────────────────────────────────────
DOMAIN_COLORS = {
    "music":   {"bg": "#EEEDFE", "border": "#3C3489"},
    "speech":  {"bg": "#E1F5EE", "border": "#085041"},
    "sfx":     {"bg": "#FAEEDA", "border": "#633806"},
    "ambient": {"bg": "#FAECE7", "border": "#712B13"},
}
DEFAULT_COLOR = {"bg": "#e8e8e6", "border": "#888"}

RAG_ON_COLOR  = "#378add"
RAG_OFF_COLOR = "#D85A30"


def _border(domain: str) -> str:
    return DOMAIN_COLORS.get(domain, DEFAULT_COLOR)["border"]


def _bg(domain: str) -> str:
    return DOMAIN_COLORS.get(domain, DEFAULT_COLOR)["bg"]


def cri_variance(scores_by_domain: dict) -> float:
    means = [float(np.mean(v)) for v in scores_by_domain.values() if v]
    if len(means) < 2:
        return 0.0
    gm = float(np.mean(means))
    return float(np.sqrt(sum((m - gm) ** 2 for m in means) / len(means)))


# ── Core batch runner ─────────────────────────────────────────────────────────

def run_batch(
    prompts_df: pd.DataFrame,
    generator: AudioGenerator,
    evaluator: AudioEvaluator,
    rag: PromptEnhancer,
    use_rag: bool,
    audio_dir: str,
) -> pd.DataFrame:
    tag     = "rag_on" if use_rag else "rag_off"
    records = []

    for idx, row in prompts_df.iterrows():
        prompt  = str(row["prompt"])
        domain  = str(row["domain"]).lower()
        gender  = str(row.get("gender",  "female")).lower()
        emotion = str(row.get("emotion", "neutral")).lower()
        steps   = int(row.get("num_inference_steps", 120))
        alen    = float(row.get("audio_length_in_s", 8.0))

        print(f"\n[{tag}] #{idx+1}/{len(prompts_df)} | domain={domain} | {prompt[:70]!r}")

        # ── Enhance (or not) ──────────────────────────────────────────────
        enhanced = rag.enhance(prompt, domain) if use_rag else prompt

        # ── Generate ──────────────────────────────────────────────────────
        run_id     = str(uuid.uuid4())[:8]
        audio_file = os.path.join(audio_dir, f"{domain}_{tag}_{run_id}.wav")

        try:
            generator.generate(
                prompt=enhanced,
                output_path=audio_file,
                domain=domain,
                gender=gender,
                emotion=emotion,
                num_inference_steps=steps,
                audio_length_in_s=alen,
            )
        except Exception as exc:
            warnings.warn(f"[Generate] FAILED for row {idx}: {exc}")
            continue

        # ── Evaluate ──────────────────────────────────────────────────────
        try:
            scores = evaluator.evaluate_poas(
                text_prompt=enhanced,       # RAG-enhanced text → CLAP
                audio_path=audio_file,
                domain=domain,
                asr_reference=prompt,       # ORIGINAL prompt → WER reference (FIX)
            )
        except Exception as exc:
            warnings.warn(f"[Evaluate] FAILED for row {idx}: {exc}")
            continue

        records.append({
            "id":               run_id,
            "domain":           domain,
            "original_prompt":  prompt,
            "enhanced_prompt":  enhanced,
            "rag_used":         use_rag,
            "gender":           gender,
            "emotion":          emotion,
            "num_steps":        steps,
            "audio_length_s":   alen,
            "poas_score":       scores["poas_score"],
            "clap_score":       scores["clap_score"],
            "semantic_score":   scores["semantic_score"],
            "transcript":       scores.get("transcript", ""),   # NEW: log transcript
            "audio_file":       audio_file,
        })
        print(f"  → POAS={scores['poas_score']:.4f}  "
              f"CLAP={scores['clap_score']:.4f}  "
              f"SEM={scores['semantic_score']:.4f}  "
              f"TRANSCRIPT={scores.get('transcript', '')!r:.50}")

    return pd.DataFrame(records)


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary(df_on: pd.DataFrame, df_off: pd.DataFrame) -> pd.DataFrame:
    def domain_stats(df):
        return (
            df.groupby("domain")["poas_score"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "mean_poas", "std": "std_poas", "count": "n"})
            .reset_index()
        )

    on_stats  = domain_stats(df_on).add_suffix("_rag_on").rename(columns={"domain_rag_on":  "domain"})
    off_stats = domain_stats(df_off).add_suffix("_rag_off").rename(columns={"domain_rag_off": "domain"})

    summary = pd.merge(on_stats, off_stats, on="domain", how="outer").fillna(0)
    summary["delta_poas"]        = (summary["mean_poas_rag_on"] - summary["mean_poas_rag_off"]).round(4)
    summary["mean_poas_rag_on"]  = summary["mean_poas_rag_on"].round(4)
    summary["mean_poas_rag_off"] = summary["mean_poas_rag_off"].round(4)
    summary["std_poas_rag_on"]   = summary["std_poas_rag_on"].round(4)
    summary["std_poas_rag_off"]  = summary["std_poas_rag_off"].round(4)

    buckets_on  = df_on.groupby("domain")["poas_score"].apply(list).to_dict()
    buckets_off = df_off.groupby("domain")["poas_score"].apply(list).to_dict()
    cri_on  = round(cri_variance(buckets_on),  4)
    cri_off = round(cri_variance(buckets_off), 4)

    totals = pd.DataFrame([{
        "domain":           "OVERALL",
        "mean_poas_rag_on":  round(df_on["poas_score"].mean(),  4),
        "std_poas_rag_on":   round(df_on["poas_score"].std(),   4),
        "n_rag_on":          len(df_on),
        "mean_poas_rag_off": round(df_off["poas_score"].mean(), 4),
        "std_poas_rag_off":  round(df_off["poas_score"].std(),  4),
        "n_rag_off":         len(df_off),
        "delta_poas":        round(df_on["poas_score"].mean() - df_off["poas_score"].mean(), 4),
        "cri_rag_on":        cri_on,
        "cri_rag_off":       cri_off,
    }])

    return pd.concat([summary, totals], ignore_index=True)


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_figure(
    df_on: pd.DataFrame,
    df_off: pd.DataFrame,
    summary: pd.DataFrame,
    out_path: str,
):
    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    fig.patch.set_facecolor("#f5f5f3")
    for ax in axes.flat:
        ax.set_facecolor("#f5f5f3")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)

    domain_rows = summary[summary["domain"] != "OVERALL"]
    domains     = list(domain_rows["domain"])
    x           = np.arange(len(domains))
    w           = 0.35

    # ── [0,0] Mean POAS ───────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.bar(x - w/2, domain_rows["mean_poas_rag_on"],  w, label="RAG on",  color=RAG_ON_COLOR,  alpha=0.82, zorder=3)
    ax.bar(x + w/2, domain_rows["mean_poas_rag_off"], w, label="RAG off", color=RAG_OFF_COLOR, alpha=0.82, zorder=3)
    ax.errorbar(x - w/2, domain_rows["mean_poas_rag_on"],  yerr=domain_rows["std_poas_rag_on"],
                fmt="none", color="#1a1a18", capsize=4, linewidth=1.2, zorder=4)
    ax.errorbar(x + w/2, domain_rows["mean_poas_rag_off"], yerr=domain_rows["std_poas_rag_off"],
                fmt="none", color="#1a1a18", capsize=4, linewidth=1.2, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(domains)
    ax.set_ylim(0, 1); ax.set_ylabel("Mean POAS", fontsize=9)
    ax.set_title("Mean POAS per domain", fontsize=10, fontweight="500")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3, zorder=0)

    # ── [0,1] Delta diverging bar ─────────────────────────────────────────
    ax = axes[0, 1]
    deltas = list(domain_rows["delta_poas"])
    colors = [RAG_ON_COLOR if d >= 0 else RAG_OFF_COLOR for d in deltas]
    ax.barh(domains, deltas, color=colors, alpha=0.82, zorder=3)
    ax.axvline(0, color="#1a1a18", linewidth=0.8, zorder=4)
    for i, (d, v) in enumerate(zip(domains, deltas)):
        ax.text(v + (0.002 if v >= 0 else -0.002), i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_xlabel("POAS delta  (RAG on − RAG off)", fontsize=9)
    ax.set_title("RAG impact per domain", fontsize=10, fontweight="500")
    ax.grid(axis="x", alpha=0.3, zorder=0)

    # ── [1,0] Per-run scatter ─────────────────────────────────────────────
    ax = axes[1, 0]
    all_runs = pd.concat([
        df_on.assign(rag_label="RAG on"),
        df_off.assign(rag_label="RAG off"),
    ]).reset_index(drop=True)
    all_runs["run_idx"] = range(1, len(all_runs) + 1)
    markers = {"RAG on": "o", "RAG off": "s"}
    for (dom, rag), grp in all_runs.groupby(["domain", "rag_label"]):
        ax.scatter(grp["run_idx"], grp["poas_score"],
                   color=_border(dom), marker=markers[rag],
                   alpha=0.85, s=55, zorder=3,
                   label=f"{dom} / {rag}" if rag == "RAG on" else None)
    handles = [mpatches.Patch(color=_border(d), label=d) for d in DOMAIN_COLORS]
    handles += [
        plt.Line2D([0],[0], marker="o", color="#555", label="RAG on",  linestyle="None", markersize=6),
        plt.Line2D([0],[0], marker="s", color="#555", label="RAG off", linestyle="None", markersize=6),
    ]
    ax.legend(handles=handles, fontsize=7, ncol=2)
    ax.set_ylim(0, 1); ax.set_ylabel("POAS", fontsize=9)
    ax.set_xlabel("Run index", fontsize=9)
    ax.set_title("Per-run POAS (● RAG on  ■ RAG off)", fontsize=10, fontweight="500")
    ax.grid(alpha=0.3, zorder=0)

    # ── [1,1] CLAP score ──────────────────────────────────────────────────
    ax = axes[1, 1]
    clap_on  = df_on.groupby("domain")["clap_score"].mean().reindex(domains).fillna(0)
    clap_off = df_off.groupby("domain")["clap_score"].mean().reindex(domains).fillna(0)
    ax.bar(x - w/2, clap_on,  w, label="RAG on",  color=RAG_ON_COLOR,  alpha=0.82, zorder=3)
    ax.bar(x + w/2, clap_off, w, label="RAG off", color=RAG_OFF_COLOR, alpha=0.82, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(domains)
    ax.set_ylim(0, 1); ax.set_ylabel("Mean CLAP score", fontsize=9)
    ax.set_title("CLAP score per domain", fontsize=10, fontweight="500")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3, zorder=0)

    # ── [2,0] Semantic — speech only ──────────────────────────────────────
    ax = axes[2, 0]
    sp_on  = df_on[df_on["domain"]  == "speech"]["semantic_score"]
    sp_off = df_off[df_off["domain"] == "speech"]["semantic_score"]
    ax.bar(["Speech\nRAG on", "Speech\nRAG off"],
           [sp_on.mean()  if len(sp_on)  else 0,
            sp_off.mean() if len(sp_off) else 0],
           color=[RAG_ON_COLOR, RAG_OFF_COLOR], alpha=0.82, width=0.4, zorder=3)
    ax.errorbar([0, 1],
                [sp_on.mean()  if len(sp_on)  else 0,
                 sp_off.mean() if len(sp_off) else 0],
                yerr=[sp_on.std()  if len(sp_on)  > 1 else 0,
                      sp_off.std() if len(sp_off) > 1 else 0],
                fmt="none", color="#1a1a18", capsize=5, linewidth=1.2, zorder=4)
    ax.set_ylim(0, 1); ax.set_ylabel("Mean semantic score (WER-based)", fontsize=9)
    ax.set_title("Semantic score — speech domain only", fontsize=10, fontweight="500")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    if len(sp_on) == 0 and len(sp_off) == 0:
        ax.text(0.5, 0.5, "No speech samples", transform=ax.transAxes,
                ha="center", va="center", color="#aaa", fontsize=10)

    # ── [2,1] CRI variance ────────────────────────────────────────────────
    ax = axes[2, 1]
    overall_row = summary[summary["domain"] == "OVERALL"]
    cri_on  = float(overall_row["cri_rag_on"].iloc[0])  if len(overall_row) else 0.0
    cri_off = float(overall_row["cri_rag_off"].iloc[0]) if len(overall_row) else 0.0
    ax.bar(["RAG on", "RAG off"], [cri_on, cri_off],
           color=[RAG_ON_COLOR, RAG_OFF_COLOR], alpha=0.82, width=0.4, zorder=3)
    for i, val in enumerate([cri_on, cri_off]):
        ax.text(i, val + 0.002, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("CRI variance", fontsize=9)
    ax.set_title("Cross-domain Robustness Index", fontsize=10, fontweight="500")
    ax.set_ylim(0, max(cri_on, cri_off) * 1.4 + 0.01)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle("Batch Evaluation — RAG On vs RAG Off",
                 fontsize=13, fontweight="500", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n[Plot] Saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch evaluate prompts with/without RAG.")
    parser.add_argument("--input",       default="prompts.csv")
    parser.add_argument("--output-dir",  default="outputs")
    parser.add_argument("--skip-rag-on",  action="store_true")
    parser.add_argument("--skip-rag-off", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    if not os.path.exists(args.input):
        sys.exit(f"[Error] Input file not found: {args.input}")
    prompts_df = pd.read_csv(args.input)
    missing = {"prompt", "domain"} - set(prompts_df.columns)
    if missing:
        sys.exit(f"[Error] Input CSV missing columns: {missing}")

    print(f"[Batch] Loaded {len(prompts_df)} prompts from {args.input}")
    print(f"[Batch] Output dir: {args.output_dir}\n")

    print("[Init] Loading models…")
    generator = AudioGenerator()
    evaluator = AudioEvaluator()
    rag       = PromptEnhancer()
    print("[Init] Models ready.\n")

    rag_on_csv  = os.path.join(args.output_dir, "results_rag_on.csv")
    rag_off_csv = os.path.join(args.output_dir, "results_rag_off.csv")

    if not args.skip_rag_on:
        print("=" * 60 + "\nPASS 1 — RAG ON\n" + "=" * 60)
        df_on = run_batch(prompts_df, generator, evaluator, rag, use_rag=True,  audio_dir=audio_dir)
        df_on.to_csv(rag_on_csv, index=False)
        print(f"\n[Saved] {rag_on_csv}  ({len(df_on)} rows)")
    else:
        print(f"[Skip] RAG-on pass — loading {rag_on_csv}")
        df_on = pd.read_csv(rag_on_csv)

    if not args.skip_rag_off:
        print("\n" + "=" * 60 + "\nPASS 2 — RAG OFF\n" + "=" * 60)
        df_off = run_batch(prompts_df, generator, evaluator, rag, use_rag=False, audio_dir=audio_dir)
        df_off.to_csv(rag_off_csv, index=False)
        print(f"\n[Saved] {rag_off_csv}  ({len(df_off)} rows)")
    else:
        print(f"[Skip] RAG-off pass — loading {rag_off_csv}")
        df_off = pd.read_csv(rag_off_csv)

    summary     = build_summary(df_on, df_off)
    summary_csv = os.path.join(args.output_dir, "comparison_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"\n[Saved] {summary_csv}")

    print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    cols = ["domain", "mean_poas_rag_on", "mean_poas_rag_off", "delta_poas"]
    print(summary[cols].to_string(index=False))
    overall = summary[summary["domain"] == "OVERALL"].iloc[0]
    print(f"\n  CRI (RAG on):  {overall.get('cri_rag_on',  0):.4f}")
    print(f"  CRI (RAG off): {overall.get('cri_rag_off', 0):.4f}")

    plot_path = os.path.join(args.output_dir, "batch_comparison.png")
    make_figure(df_on, df_off, summary, plot_path)

    print(f"\n[Done] All outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
