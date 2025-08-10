"""
api/routes.py
APIRouter only. Keep app-level startup/middleware in main.py.
"""

from fastapi import (
    APIRouter, File, UploadFile, Depends, HTTPException, status, Query, Form
)
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Any, Dict
import pandas as pd
import io
import os

from ingestion.ingestion_service import parse_files


    


# Local imports from your package
# from ingestion.parsers import parse_xxx   <-- later replace local parsing with ingestion module
# from query_engine.query_executor import QueryExecutor

router = APIRouter(tags=["amma"])

# ---------- Security stub (replace with real auth in production) ----------
security = HTTPBasic()


def get_current_user(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """Simple basic-auth stub. Replace with JWT/OAuth in prod."""
    if credentials.username != os.getenv("ADMIN_USER", "admin") or credentials.password != os.getenv("ADMIN_PASS", "password"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ---------- Schemas ----------
class IngestionResponse(BaseModel):
    message: str
    columns: List[str]
    preview: List[Dict[str, Any]]


class CleanRequest(BaseModel):
    columns: Optional[List[str]] = Field(None)
    drop_duplicates: Optional[bool] = Field(False)
    fillna: Optional[Any] = Field(None)


class AnalysisRequest(BaseModel):
    operations: List[str]
    columns: Optional[List[str]] = None


class ModelRequest(BaseModel):
    model_type: str
    target: str
    features: Optional[List[str]] = None
    params: Optional[Dict[str, Any]] = None


class SQLExportRequest(BaseModel):
    table_name: str
    dialect: str
    primary_keys: Optional[List[str]] = None


class QueryRequest(BaseModel):
    sql: str
    params: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    detail: str


class PaginatedResponse(BaseModel):
    items: List[Dict[str, Any]]
    total: int
    page: int
    size: int


# ---------- Helpers ----------
async def parse_uploaded_file(file: UploadFile) -> pd.DataFrame:
    """
    Async file -> pandas DataFrame.
    Supports CSV and Excel. For more formats, move logic to ingestion/parsers.py
    """
    try:
        content = await file.read()
        # small defensive check
        if not file.filename:
            raise ValueError("Uploaded file must have a filename.")
        lower = file.filename.lower()
        if lower.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif lower.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            # In production: call ingestion/file_detector + parsers for more formats
            raise ValueError("Only CSV and Excel are supported by this endpoint.")
        return df
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File parsing error: {e}")


def sanitize_sql(sql: str) -> str:
    """Light check. Always use parameterized queries in QueryExecutor."""
    if not sql or len(sql.strip()) == 0:
        raise HTTPException(status_code=400, detail="Empty SQL")
    dangerous = [";--", "/*", "*/", "xp_"]
    lower = sql.lower()
    for d in dangerous:
        if d in lower:
            raise HTTPException(status_code=400, detail="Potentially unsafe SQL detected.")
    return sql


# ---------- Endpoints ----------
@router.post(
    "/ingest",
    response_model=List[IngestionResponse],
    responses={400: {"model": ErrorResponse}},
    dependencies=[Depends(RateLimiter(times=5, seconds=60))],
)
async def ingest_file(files: List[UploadFile] = File(...), user: str = Depends(get_current_user)):
    parsed_files = await parse_files(files)
    results = []
    for filename, df in parsed_files:
        preview = df.head(10).to_dict(orient="records")
        results.append(IngestionResponse(
            message=f"{filename} ingested successfully.",
            columns=list(df.columns),
            preview=preview
        ))
    return results
    

@router.post(
    "/clean",
    response_model=IngestionResponse,
    responses={400: {"model": ErrorResponse}},
    dependencies=[Depends(RateLimiter(times=10, seconds=60))],
)
async def clean_data(
    file: UploadFile = File(...),
    clean_req: CleanRequest = Depends(),
    user: str = Depends(get_current_user),
):
    """Simple cleaning wrapper — put heavy logic in cleaning_pipeline.py and call here or as background task."""
    df = await parse_uploaded_file(file)
    if clean_req.columns:
        # validate columns exist
        missing = [c for c in clean_req.columns if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Columns not found: {missing}")
        df = df[clean_req.columns]
    if clean_req.drop_duplicates:
        df = df.drop_duplicates()
    if clean_req.fillna is not None:
        df = df.fillna(clean_req.fillna)
    preview = df.head(10).to_dict(orient="records")
    return IngestionResponse(message="Data cleaned successfully.", columns=list(df.columns), preview=preview)


@router.post(
    "/analyze",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}},
    dependencies=[Depends(RateLimiter(times=10, seconds=60))],
)
async def analyze_data(file: UploadFile = File(...), analysis_req: AnalysisRequest = Depends(), user: str = Depends(get_current_user)):
    """Profiling and simple analysis. Production: call analysis/statistical_profiler.py"""
    df = await parse_uploaded_file(file)
    ops = analysis_req.operations
    results: Dict[str, Any] = {}
    if "describe" in ops:
        results["describe"] = df.describe(include="all").to_dict()
    if "correlation" in ops and df.select_dtypes(include=["number"]).shape[1] > 1:
        results["correlation"] = df.corr().to_dict()
    return results


@router.post(
    "/model",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}},
    dependencies=[Depends(RateLimiter(times=5, seconds=60))],
)
async def model_data(file: UploadFile = File(...), model_req: ModelRequest = Depends(), user: str = Depends(get_current_user)):
    """
    Small example. Production: hand off to modeling/forecasting modules and return job id.
    """
    df = await parse_uploaded_file(file)
    from sklearn.linear_model import LinearRegression  # keep local import
    features = model_req.features or [c for c in df.columns if c != model_req.target]
    if model_req.target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{model_req.target}' not found")
    X = df[features]
    y = df[model_req.target]
    if model_req.model_type == "linear_regression":
        model = LinearRegression(**(model_req.params or {}))
        model.fit(X, y)
        preds = model.predict(X)
        return {"message": "Model fitted.", "coefficients": model.coef_.tolist(), "intercept": model.intercept_, "predictions": preds[:10].tolist()}
    raise HTTPException(status_code=400, detail="Only 'linear_regression' is implemented.")


@router.post(
    "/sql/export",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}},
    dependencies=[Depends(RateLimiter(times=5, seconds=60))],
)
async def sql_export(file: UploadFile = File(...), sql_req: SQLExportRequest = Depends(), user: str = Depends(get_current_user)):
    """
    Generate DDL. In prod, call sql_export.schema_inference & ddl_generator.
    """
    df = await parse_uploaded_file(file)
    # Use your sql_export module (example, will error if missing)
    from sql_export.schema_inference import infer_schema
    from sql_export.ddl_generator import DDLGenerator
    schema = infer_schema(df, table_name=sql_req.table_name, key_overrides=sql_req.primary_keys)
    ddlgen = DDLGenerator(dialect=sql_req.dialect)
    statement = ddlgen.generate_create_table(schema)
    return {"ddl": statement}


@router.post(
    "/sql/query",
    response_model=PaginatedResponse,
    responses={400: {"model": ErrorResponse}},
    dependencies=[Depends(RateLimiter(times=10, seconds=60))],
)
async def sql_query(query_req: QueryRequest, page: int = Query(1, ge=1), size: int = Query(100, ge=1, le=1000), user: str = Depends(get_current_user)):
    """
    Execute SQL safely via QueryExecutor. QueryExecutor must implement parameterized execution.
    """
    sql = sanitize_sql(query_req.sql)
    params = query_req.params or {}
    # Use QueryExecutor from your query_engine module
    from query_engine.query_executor import QueryExecutor
    db_url = os.getenv("DATABASE_URL", "sqlite:///data.db")
    executor = QueryExecutor(db_url)
    df = executor.execute(sql, params)  # must return pandas.DataFrame
    total = len(df)
    start = (page - 1) * size
    end = start + size
    items = df.iloc[start:end].to_dict(orient="records")
    return PaginatedResponse(items=items, total=total, page=page, size=size)

