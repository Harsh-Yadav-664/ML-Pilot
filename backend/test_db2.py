import asyncio
from app.db.session import AsyncSessionLocal
from app.services.experiment_service import ExperimentService

async def run():
    async with AsyncSessionLocal() as db:
        svc = ExperimentService(db)
        exps, total = await svc.list_by_project('demo-project-id')
        print('Total:', total)
        for e in exps:
            print(e.id, e.status, e.model_name)

asyncio.run(run())
