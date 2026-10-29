"""LangGraph CLI entry point; use LangSmith Studio to interact with the graph."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from starlette.applications import Starlette

from parking_agent import ParkingSpaceReservationAgent

parking_agent = ParkingSpaceReservationAgent(require_startup=True)
agent = parking_agent.graph


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncGenerator[None, None]:
    try:
        await asyncio.to_thread(parking_agent.prepare_startup)
        yield
    finally:
        # Also invalidate readiness if startup is cancelled while its worker finishes.
        await asyncio.to_thread(parking_agent.finish_shutdown)


# Module imports in langgraph.json ensure the graph and lifespan share this instance.
# No public routes/UI are added; Agent Server supplies its usual APIs and Studio.
webapp = Starlette(lifespan=lifespan)
