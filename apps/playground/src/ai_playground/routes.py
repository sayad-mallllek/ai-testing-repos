from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai_playground.database import get_session
from ai_playground.models import Run
from ai_playground.schemas import RunCreate, RunRead
from ai_playground.service import run_experiment

router = APIRouter(prefix="/runs", tags=["experiments"])
SessionDependency = Annotated[AsyncSession, Depends(get_session)]


@router.post("", response_model=RunRead, status_code=201)
async def create_run(payload: RunCreate, session: SessionDependency):
    output = await run_experiment(payload.prompt, payload.parameters)
    run = Run(prompt=payload.prompt, parameters=payload.parameters, output=output)
    session.add(run)
    await session.commit()
    await session.refresh(run)
    return run


@router.get("", response_model=list[RunRead])
async def list_runs(
    session: SessionDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    result = await session.scalars(
        select(Run).order_by(Run.created_at.desc(), Run.id.desc()).limit(limit).offset(offset)
    )
    return result.all()


@router.get("/{run_id}", response_model=RunRead)
async def get_run(run_id: UUID, session: SessionDependency):
    run = await session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run
