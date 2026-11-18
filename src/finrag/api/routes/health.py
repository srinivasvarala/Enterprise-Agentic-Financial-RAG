"""Liveness endpoint. Dependency readiness checks arrive with M1."""

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from finrag import __version__

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    env: str


@router.get("/health")
def health(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, env=request.app.state.settings.env)
