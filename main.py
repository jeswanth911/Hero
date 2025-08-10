# main.py
from fastapi import FastAPI
from api import routes as api_router
app = api_router.app

from api.logging import setup_logging  # optional; create a simple function if not present

def create_app() -> FastAPI:
    app = FastAPI(
        title="AMMA AI Analyst",
        version="0.1.0",
        description="AMMA: multi-file ingestion -> cleaning -> SQL -> NL queries"
    )

    # optional logging setup if you implemented api.logging
    try:
        setup_logging()
    except Exception:
        pass

    # include API router
    app.include_router(api_router, prefix="/api")

    # simple root /health
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
  
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router

def create_app() -> FastAPI:
    app = FastAPI(title="AMMA AI Analyst", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api")
    return app

app = create_app()
