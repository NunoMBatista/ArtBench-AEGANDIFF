import os
import wandb
import json
import pandas as pd
import dotenv
import datetime
from collections.abc import Mapping


def _json_default(obj):
    """Best-effort conversion of common non-JSON-serializable objects."""
    # Datetime-like
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()

    # Mapping-like objects (e.g., wandb SummarySubDict)
    if isinstance(obj, Mapping):
        return dict(obj)

    # Sets -> lists
    if isinstance(obj, set):
        return list(obj)

    # Numpy scalars/arrays (optional dependency)
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # Fallback: stringify unknown objects
    return str(obj)

def fetch_wandb_data():
    api = wandb.Api()
    
    dotenv.load_dotenv()
    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")
    
    # entity = os.getenv("WANDB_ENTITY")
    # project = os.getenv("WANDB_PROJECT")

    if not entity or not project:
        print("WANDB_ENTITY or WANDB_PROJECT not set.")
        return

    full_path = f"{entity}/{project}"
    print(f"Fetching runs from {full_path}...")
    
    print("Initializing WandB API...")
    runs = api.runs(full_path)
    

    run_list = []
    for run in runs:
        print(f"Processing run: {run.name} (ID: {run.id})")
        # Fetching run details
        run_data = {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "tags": run.tags,
            "config": run.config,
            "summary": dict(run.summary),
            "created_at": run.created_at,
        }
        run_list.append(run_data)
        
    # Save to JSON for structured analysis
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/wandb_all_runs.json", "w") as f:
        json.dump(run_list, f, indent=2, default=_json_default)
        
    print(f"Successfully fetched {len(run_list)} runs and saved to outputs/wandb_all_runs.json")

if __name__ == "__main__":
    fetch_wandb_data()
