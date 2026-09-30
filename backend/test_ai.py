import asyncio
from app.core.config import settings
from ai.gateway import AIGateway
from ml.experiments.planner import ExperimentPlanner
from ml.data.profiling.profiler import ProfileResult
import logging

logging.basicConfig(level=logging.INFO)

async def test():
    gw = AIGateway(settings)
    print("Providers:", gw.list_providers())
    p = ExperimentPlanner(gw)
    pr = ProfileResult(rows=100, columns=5, missing_rate=0.1, duplicate_rows=0, target_balance={}, column_stats={}, warnings=[])
    try:
        res = await p.generate_hypotheses(pr, "Survived", "Maximize F1")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(test())
