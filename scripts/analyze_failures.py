import pandas as pd
import argparse
import os
import sys

def analyze():
    parser = argparse.ArgumentParser(description="Analyze Prompt Evaluation Failures")
    parser.add_argument("--results", default="results/results.csv", help="Path to results CSV")
    parser.add_argument("--threshold", type=float, default=5.0, help="Score threshold for failure detection")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"File not found: {args.results}")
        return

    print(f"🔍 Analyzing Failures in {args.results} (Threshold < {args.threshold})")
    
    df = pd.read_csv(args.results)
    
    # Identify score columns
    score_cols = [c for c in df.columns if c.startswith("score_")]
    
    failures = []
    
    for _, row in df.iterrows():
        failed_criteria = []
        for col in score_cols:
            if row[col] < args.threshold:
                failed_criteria.append(f"{col} ({row[col]})")
        
        if failed_criteria:
            failures.append({
                "model": row["model"],
                "query": row["query"],
                "response": row["response"],
                "failures": ", ".join(failed_criteria)
            })
            
    if not failures:
        print("✅ No failures detected above threshold.")
        return

    print(f"⚠️ Found {len(failures)} failures:\n")
    
    for f in failures:
        print(f"❌ Model: {f['model']}")
        print(f"   Query: {f['query']}")
        print(f"   Failures: {f['failures']}")
        print(f"   Response Preview: {str(f['response'])[:100]}...")
        print("-" * 50)
        
    # Categorization logic (simple keyword based)
    print("\n📊 Failure Categorization (Heuristic):")
    cats = {"Refusal": 0, "Hallucination": 0, "Empty": 0, "Other": 0}
    
    for f in failures:
        resp = str(f['response']).lower()
        if "cannot" in resp or "sorry" in resp:
            cats["Refusal"] += 1
        elif len(resp) < 5:
            cats["Empty"] += 1
        # Hallucination is hard to detect without ground truth match failure, 
        # but let's assume low accuracy score + high confidence (not implemented) 
        # is hallucination. For now just "Other".
        else:
            cats["Other"] += 1
            
    print(cats)

if __name__ == "__main__":
    analyze()
