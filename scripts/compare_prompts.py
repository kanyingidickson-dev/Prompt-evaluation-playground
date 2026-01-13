import pandas as pd
import argparse
import os

def compare():
    parser = argparse.ArgumentParser(description="Compare Prompt Performance")
    parser.add_argument("--results", default="results/results.csv", help="Path to results CSV")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"File not found: {args.results}")
        return

    print(f"⚖️  Comparing Prompts in {args.results}")
    
    df = pd.read_csv(args.results)
    
    # Check if we have multiple prompts
    if "prompt_source" not in df.columns:
        print("Error: 'prompt_source' column missing. Cannot compare prompts.")
        return

    # Group by Prompt Source and Model
    score_cols = [c for c in df.columns if c.startswith("score_")]
    
    grouped = df.groupby(["prompt_source", "model"])[score_cols].mean()
    print("\n📈 Mean Scores by Prompt & Model:")
    print(grouped)
    
    # Ranking logic
    # Calculate weighted score if config provided? For now just average of scores
    df['overall_score'] = df[score_cols].mean(axis=1)
    
    ranking = df.groupby("prompt_source")['overall_score'].mean().sort_values(ascending=False)
    print("\n🏆 Prompt Ranking (Overall Score):")
    print(ranking)
    
    best_prompt = ranking.index[0]
    print(f"\n✨ Best Performing Prompt: {best_prompt}")

if __name__ == "__main__":
    compare()
