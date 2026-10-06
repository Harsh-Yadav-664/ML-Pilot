"""Projects API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.deps import DBSession, OwnerID
from app.schemas.common import PaginatedResponse
from app.schemas.project import ProjectCreate, ProjectRead, ProjectUpdate
from app.services.project_service import ProjectService

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("/", response_model=ProjectRead, status_code=status.HTTP_201_CREATED)
async def create_project(data: ProjectCreate, db: DBSession, owner_id: OwnerID) -> ProjectRead:
    svc = ProjectService(db)
    project = await svc.create(data, owner_id)
    return ProjectRead.model_validate(project)


@router.get("/", response_model=PaginatedResponse[ProjectRead])
async def list_projects(
    db: DBSession,
    owner_id: OwnerID,
    page: int = 1,
    page_size: int = 20,
) -> PaginatedResponse[ProjectRead]:
    svc = ProjectService(db)
    projects, total = await svc.list_by_owner(owner_id, page, page_size)
    return PaginatedResponse(
        items=[ProjectRead.model_validate(p) for p in projects],
        total=total,
        page=page,
        page_size=page_size,
        has_next=(page * page_size) < total,
        has_prev=page > 1,
    )


@router.get("/{project_id}", response_model=ProjectRead)
async def get_project(project_id: str, db: DBSession) -> ProjectRead:
    svc = ProjectService(db)
    project = await svc.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectRead.model_validate(project)


@router.patch("/{project_id}", response_model=ProjectRead)
async def update_project(project_id: str, data: ProjectUpdate, db: DBSession) -> ProjectRead:
    svc = ProjectService(db)
    project = await svc.update(project_id, data)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectRead.model_validate(project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: str, db: DBSession) -> None:
    svc = ProjectService(db)
    deleted = await svc.delete(project_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")
