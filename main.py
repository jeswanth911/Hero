# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.routes import router as api_router
from pydantic import ValidationError


def create_app() -> FastAPI:
    app = FastAPI(
        title="AMMA AI Analyst",
        version="1.0.0",
        description="Multi-file ingestion → cleaning → SQL → natural language queries"
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Change in prod
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request, exc):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    # Routes
    app.include_router(api_router, prefix="/api")

    # Health
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    
