"""FastAPI application setup for ComicCraft.

Only application wiring lives here. Business logic must live
in dedicated service modules, not in this file.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routes import router

BASE_DIR = Path(__file__).resolve().parent.parent


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="ComicCraft")

    app.mount(
        "/static",
        StaticFiles(directory=BASE_DIR / "static"),
        name="static",
    )

    templates = Jinja2Templates(directory=BASE_DIR / "templates")
    app.state.templates = templates

    app.include_router(router)

    return app


app = create_app()
