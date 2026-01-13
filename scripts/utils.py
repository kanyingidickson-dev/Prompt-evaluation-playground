import yaml
import json
import os
from typing import List, Dict, Any

def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_prompt(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()

def load_dataset(path: str) -> List[Dict[str, str]]:
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_results(results: List[Dict[str, Any]], output_dir: str, format: str = "json"):
    os.makedirs(output_dir, exist_ok=True)
    
    if format == "json":
        with open(os.path.join(output_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
    
    # Simple CSV export
    if format == "csv" or True: # always do CSV too
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
