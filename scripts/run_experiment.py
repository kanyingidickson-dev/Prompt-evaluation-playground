import sys
import os
import argparse
import time
from typing import List, Dict
import pandas as pd

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils import load_config, load_prompt, load_dataset, save_results
from models.openai_client import OpenAIClient
from models.local_model_client import LocalModelClient
from evaluators.relevance import RelevanceEvaluator
from evaluators.safety import SafetyEvaluator
from evaluators.accuracy import AccuracyEvaluator
from evaluators.clarity import ClarityEvaluator

def get_model(model_conf: Dict):
    provider = model_conf.get("provider")
    if provider == "openai":
        return OpenAIClient(model_conf)
    elif provider == "local":
        return LocalModelClient(model_conf)
    else:
        raise ValueError(f"Unknown model provider: {provider}")

def get_evaluators(eval_conf: Dict) -> Dict:
    evaluators = {}
    if eval_conf.get("relevance", {}).get("enabled"):
        evaluators["relevance"] = RelevanceEvaluator(eval_conf["relevance"])
    if eval_conf.get("safety", {}).get("enabled"):
        evaluators["safety"] = SafetyEvaluator(eval_conf["safety"])
    if eval_conf.get("accuracy", {}).get("enabled"):
        evaluators["accuracy"] = AccuracyEvaluator(eval_conf["accuracy"])
    if eval_conf.get("clarity", {}).get("enabled"):
        evaluators["clarity"] = ClarityEvaluator(eval_conf["clarity"])
    return evaluators

def run():
    parser = argparse.ArgumentParser(description="Run Prompt Evaluation Experiment")
    parser.add_argument("--config", default="config/evaluation.yaml", help="Path to evaluation config")
    parser.add_argument("--models-config", default="config/models.yaml", help="Path to models config")
    args = parser.parse_args()

    # Load Configs
    eval_config = load_config(args.config)
    models_config = load_config(args.models_config)

    print(f"🚀 Starting Experiment: {eval_config.get('experiment_name')}")
    
    # Initialize Evaluators
    evaluators = get_evaluators(eval_config.get("evaluators", {}))
    print(f"📋 Loaded Evaluators: {list(evaluators.keys())}")

    # Load Data
    prompts = [load_prompt(p) for p in eval_config["prompts"]]
    datasets = []
    for d in eval_config["datasets"]:
        datasets.extend(load_dataset(d))
    
    print(f"📝 Loaded {len(prompts)} prompts and {len(datasets)} queries.")

    results = []

    # Iterate through models defined in evaluation config that are present in models.yaml
    target_models = eval_config["models"]
    
    for model_key in target_models:
        if model_key not in models_config["models"]:
            print(f"Warning: Model {model_key} not found in models.yaml")
            continue
            
        model_conf = models_config["models"][model_key]
        model = get_model(model_conf)
        print(f"\nrunning model: {model.model_name}...")

        for prompt_template in prompts:
            for idx, item in enumerate(datasets):
                query = item["query"]
                reference = item.get("reference_answer")
                
                # Render Prompt
                rendered_prompt = prompt_template.replace("{{query}}", query)
                
                # Generate
                print(f"  generating {idx+1}/{len(datasets)}: {query[:30]}...")
                response_obj = model.generate(rendered_prompt)
                
                # Evaluate
                row = {
                    "model": model.model_name,
                    "prompt_source": os.path.basename(eval_config["prompts"][prompts.index(prompt_template)]),
                    "query": query,
                    "reference": reference,
                    "response": response_obj.content,
                    "latency_ms": response_obj.latency_ms,
                    "prompt_tokens": response_obj.token_usage.prompt_tokens,
                    "completion_tokens": response_obj.token_usage.completion_tokens,
                }
                
                for ev_name, evaluator in evaluators.items():
                    eval_res = evaluator.evaluate(query, response_obj.content, reference)
                    row[f"score_{ev_name}"] = eval_res.score
                    row[f"reason_{ev_name}"] = eval_res.reasoning

                results.append(row)

    # Save Results
    output_dir = eval_config["output"]["save_dir"]
    save_results(results, output_dir, eval_config["output"]["format"])
    
    print(f"\n✅ Experiment Complete. Results saved to {output_dir}/")
    
    # Simple summary to stdout
    df = pd.DataFrame(results)
    score_cols = [c for c in df.columns if c.startswith("score_")]
    if score_cols:
        print("\n📊 Summary Statistics:")
        print(df.groupby("model")[score_cols].mean())

if __name__ == "__main__":
    run()
