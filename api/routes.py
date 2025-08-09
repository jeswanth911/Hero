from fastapi import (
    FastAPI, File, UploadFile, Depends, HTTPException, status, Query, Form, Request
)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Any, Dict
import pandas as pd
import io
import uvicorn
import asyncio
import logging

# Example: Use Redis for rate limiting (setup not shown in this file)
import aioredis

# ========== Application & Middleware Setup ==========

app = FastAPI(
    title="Comprehensive Data Platform API",
    description="RESTful API for data ingestion, cleaning, analysis, modeling, SQL export, and query execution.",
    version="1.0.0"
)

# Allow CORS (customize origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Security (Basic Auth Example) ==========

security = HTTPBasic()

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    # Placeholder authentication logic
    if credentials.username != "admin" or credentials.password != "password":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# ========== Pydantic Schemas ==========

class IngestionResponse(BaseModel):
    message: str
    columns: List[str]
    preview: List[Dict[str, Any]]

class CleanRequest(BaseModel):
    columns: Optional[List[str]] = Field(None, description="Columns to clean")
    drop_duplicates: Optional[bool] = Field(False, description="Remove duplicate rows")
    fillna: Optional[Any] = Field(None, description="Value to fill NA values")

class AnalysisRequest(BaseModel):
    operations: List[str] = Field(..., description="List of analysis operations, e.g., ['describe', 'correlation']")
    columns: Optional[List[str]] = None

class ModelRequest(BaseModel):
    model_type: str = Field(..., description="Model type, e.g., 'linear_regression', 'random_forest'")
    target: str = Field(..., description="Target column name")
    features: Optional[List[str]] = None
    params: Optional[Dict[str, Any]] = None

class SQLExportRequest(BaseModel):
    table_name: str
    dialect: str = Field(..., description="SQL dialect, e.g., 'postgresql', 'mysql', 'sqlite'")
    primary_keys: Optional[List[str]] = None

class QueryRequest(BaseModel):
    sql: str = Field(..., description="SQL query to execute")
    params: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    detail: str

class PaginatedResponse(BaseModel):
    items: List[Dict[str, Any]]
    total: int
    page: int
    size: int

# ========== Dependency: Rate Limiting ==========

@app.on_event("startup")
async def startup():
    # Connect to Redis for rate limiting
    redis = aioredis.from_url("redis://localhost", encoding="utf8", decode_responses=True)
    await FastAPILimiter.init(redis)

# ========== Helper Functions ==========

def parse_uploaded_file(file: UploadFile) -> pd.DataFrame:
    """Parse uploaded file into a pandas DataFrame."""
    try:
        content = file.file.read()
        # Support CSV and Excel
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise ValueError("Only CSV or Excel files are supported.")
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File parsing error: {str(e)}")

def sanitize_sql(sql: str) -> str:
    """Very basic SQL sanitization."""
    # Real-world: Use parameterized queries, not string replacement.
    dangerous = [';--', '--', '/*', '*/', 'xp_']
    for d in dangerous:
        if d in sql.lower():
            raise HTTPException(status_code=400, detail="Potentially unsafe SQL detected.")
    return sql

# ========== API Endpoints ==========

@app.post(
    "/ingest",
    response_model=IngestionResponse,
    responses={400: {"model": ErrorResponse}},
    dependencies=[Depends(RateLimiter(times=5, seconds=60))]
)
async def ingest_file(
    file: UploadFile = File(...),
    user: str = Depends(get_current_user)
):
    """
    Ingest a CSV or Excel file, returning metadata and preview.
    """
    df = parse_uploaded_file(file)
    preview = df.head(10).to_dict(orient="records")
    return IngestionResponse(
        message="File ingested successfully.",
        columns=list(df.columns),
        preview=preview
    )

@app.post(
    "/clean",
    response_model=IngestionResponse,
    responses={400: {"model": ErrorResponse}},
    dependencies=[Depends(RateLimiter(times=10, seconds=60))]
)
async def clean_data(
    file: UploadFile = File(...),
    clean_req: CleanRequest = Depends(),
    user: str = Depends(get_current_user)
):
    """
    Clean uploaded data with specified options.
    """
    df = parse_uploaded_file(file)
    if clean_req.columns:
        df = df[clean_req.columns]
    if clean_req.drop_duplicates:
        df = df.drop_duplicates()
    if clean_req.fillna is not None:
        df = df.fillna(clean_req.fillna)
    preview = df.head(10).to_dict(orient="records")
    return IngestionResponse(
        message="Data cleaned successfully.",
        columns=list(df.columns),
        preview=preview
    )

@app.post(
    "/analyze",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}},
    dependencies=[Depends(RateLimiter(times=10, seconds=60))]
)
async def analyze_data(
    file: UploadFile = File(...),
    analysis_req: AnalysisRequest = Depends(),
    user: str = Depends(get_current_user)
):
    """
    Perform analysis operations on uploaded data.
    """
    df = parse_uploaded_file(file)
    ops = analysis_req.operations
    results = {}
    if "describe" in ops:
        results["describe"] = df.describe(include="all").to_dict()
    if "correlation" in ops and df.select_dtypes(include=['number']).shape[1] > 1:
        results["correlation"] = df.corr().to_dict()
    # Extend with more ops as needed
    return results

@app.post(
    "/model",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}},
    dependencies=[Depends(RateLimiter(times=5, seconds=60))]
)
async def model_data(
    file: UploadFile = File(...),
    model_req: ModelRequest = Depends(),
    user: str = Depends(get_current_user)
):
    """
    Fit a machine learning model to uploaded data.
    """
    df = parse_uploaded_file(file)
    # Example: Only basic logic here, real logic in separate module
    from sklearn.linear_model import LinearRegression
    features = model_req.features or [c for c in df.columns if c != model_req.target]
    X = df[features]
    y = df[model_req.target]
    if model_req.model_type == "linear_regression":
        model = LinearRegression(**(model_req.params or {}))
        model.fit(X, y)
        preds = model.predict(X)
        return {"message": "Model fitted.", "coefficients": model.coef_.tolist(), "intercept": model.intercept_, "predictions": preds[:10].tolist()}
    else:
        raise HTTPException(status_code=400, detail="Only 'linear_regression' is implemented in this example.")

@app.post(
    "/sql/export",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}},
    dependencies=[Depends(RateLimiter(times=5, seconds=60))]
)
async def sql_export(
    file: UploadFile = File(...),
    sql_req: SQLExportRequest = Depends(),
    user: str = Depends(get_current_user)
):
    """
    Generate SQL CREATE TABLE statement from uploaded data.
    """
    df = parse_uploaded_file(file)
    # Use schema_inference.py and ddl_generator.py here
    from schema_inference import infer_schema
    from ddl_generator import DDLGenerator
    schema = infer_schema(df, table_name=sql_req.table_name, key_overrides=sql_req.primary_keys)
    ddlgen = DDLGenerator(dialect=sql_req.dialect)
    statement = ddlgen.generate_create_table(schema)
    return {"ddl": statement}

@app.post(
    "/sql/query",
    response_model=PaginatedResponse,
    responses={400: {"model": ErrorResponse}},
    dependencies=[Depends(RateLimiter(times=10, seconds=60))]
)
async def sql_query(
    query_req: QueryRequest,
    page: int = Query(1, ge=1),
    size: int = Query(100, ge=1, le=1000),
    user: str = Depends(get_current_user)
):
    """
    Execute a SQL query and return paginated results.
    """
    sql = sanitize_sql(query_req.sql)
    params = query_req.params or {}
    # Use query_executor.py here
    from query_executor import QueryExecutor
    # db_url would come from config/environment
    db_url = "sqlite:///data.db"  # Example only
    executor = QueryExecutor(db_url)
    df = executor.execute(sql, params)
    total = len(df)
    start = (page - 1) * size
    end = start + size
    items = df.iloc[start:end].to_dict(orient="records")
    return PaginatedResponse(items=items, total=total, page=page, size=size)

# ========== Error Handling ==========

@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# ========== Run Uvicorn (for local dev) ==========

if __name__ == "__main__":
    uvicorn.run("routes:app", host="0.0.0.0", port=8080, reload=True)
