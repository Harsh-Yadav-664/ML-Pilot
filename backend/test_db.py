import asyncio
from app.db.session import AsyncSessionLocal
from app.services.experiment_service import ExperimentService
from app.schemas.experiment import ExperimentCreate

async def run():
    async with AsyncSessionLocal() as db:
        svc = ExperimentService(db)
        exp_create = ExperimentCreate(
            project_id='demo-project-id',
            dataset_version='test.csv',
            hypothesis='test',
            change_description='test',
            model_name='XGBClassifier',
            parameters={'target_column': 'target'}
        )
        try:
            exp = await svc.create(exp_create)
            print('CREATED:', exp.id)
            await db.commit()
        except Exception as e:
            import traceback
            traceback.print_exc()

asyncio.run(run())
