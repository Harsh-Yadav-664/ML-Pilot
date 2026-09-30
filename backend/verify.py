import asyncio
import sys
sys.path.insert(0, '.')

async def test():
    from app.services.experiment_service import ExperimentService
    from ml.agents.decision_agent import DecisionAgent
    from app.db.session import AsyncSessionLocal
    from app.db.models.experiment import Experiment
    from sqlalchemy import select

    # Test 1: service with None session (uses own session internally)
    svc = ExperimentService(None)
    print('OK: ExperimentService(None) created')

    # Test 2: DecisionAgent import
    from ai.gateway import AIGateway
    from app.core.config import settings
    gw = AIGateway(settings)
    agent = DecisionAgent(gw, settings)
    print('OK: DecisionAgent created')

    # Test 3: DB read still works with session
    async with AsyncSessionLocal() as db:
        svc2 = ExperimentService(db)
        exps, total = await svc2.list_by_project('demo-project-id')
        print(f'OK: list_by_project returned {total} experiments')

asyncio.run(test())
