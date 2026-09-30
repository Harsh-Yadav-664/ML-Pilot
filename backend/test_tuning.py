import asyncio
import pandas as pd
from ml.experiments.schema import ExperimentSpec
from ml.experiments.executor import LocalExperimentExecutor

def load_data(version):
    return pd.read_csv("titanic.csv")

async def main():
    executor = LocalExperimentExecutor(data_loader_func=load_data)
    
    spec = ExperimentSpec(
        id="test-exp-1",
        project_id="test-project",
        hypothesis="test tuning",
        change_description="test",
        model_name="XGBClassifier",
        dataset_version="v1",
        parameters={"target_column": "Survived", "n_trials": 2, "ensemble": True},
        validation_config={}
    )
    
    print("Running experiment...")
    result = await executor.run(spec)
    
    print("Metrics:")
    for k, v in result.metrics.items():
        print(f"  {k}: {v}")
        
    print("Best params:", result.parameters.get("best_params"))
    if "ensemble_f1" in result.metrics:
        print("Success: ensemble_f1 is in metrics.")
    else:
        print("Failed: ensemble_f1 not found.")

if __name__ == "__main__":
    asyncio.run(main())
