import asyncio
from app.db.session import AsyncSessionLocal
from app.services.experiment_service import ExperimentService
from app.schemas.experiment import ExperimentCreate, ExperimentRead

async def test():
    async with AsyncSessionLocal() as db:
        svc = ExperimentService(db)
        exp = await svc.create(ExperimentCreate(project_id='test', dataset_version='test', hypothesis='test', change_description='test', model_name='test'))
        try:
            print(ExperimentRead.model_validate(exp))
        except Exception as e:
            print('VALIDATION ERROR:', e)

asyncio.run(test())
