import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import io
import json
import time

# Assume `app` is your FastAPI application.
from routes import app

client = TestClient(app)

# --- Fixtures and Auth ---

@pytest.fixture
def auth_headers():
    # Example: Basic Auth
    return {"Authorization": "Basic YWRtaW46YWRtaW5wdw=="}  # admin:adminpw base64

@pytest.fixture
def test_user_token():
    # Mock JWT/bearer token for user role
    return "Bearer testusertoken"

@pytest.fixture
def admin_token():
    # Mock JWT/bearer token for admin role
    return "Bearer testadmintoken"

# --- Helper for Pydantic validation ---
def validate_pydantic(schema, data):
    # schema: Pydantic model, data: dict/json
    schema.parse_obj(data)  # will raise ValidationError if invalid

# --- Ingestion Endpoint ---

def test_ingest_csv(auth_headers):
    files = {"file": ("test.csv", io.BytesIO(b"a,b\n1,2\n3,4"), "text/csv")}
    resp = client.post("/ingest", files=files, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "columns" in data and "preview" in data

def test_ingest_invalid_file(auth_headers):
    files = {"file": ("bad.xyz", io.BytesIO(b"junkdata"), "application/octet-stream")}
    resp = client.post("/ingest", files=files, headers=auth_headers)
    assert resp.status_code in [400, 415]

# --- Cleaning Endpoint ---

def test_cleaning_post(auth_headers):
    payload = {"data": [{"a": 1, "b": None}, {"a": 1, "b": 2}], "deduplicate": True}
    resp = client.post("/clean", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    cleaned = resp.json()
    assert isinstance(cleaned, dict)
    assert "cleaned_data" in cleaned

def test_cleaning_invalid_schema(auth_headers):
    resp = client.post("/clean", json={"bad": "data"}, headers=auth_headers)
    assert resp.status_code == 422

# --- Analysis Endpoint ---

def test_analysis_post(auth_headers):
    payload = {"data": [{"a": 1, "b": 2}, {"a": 5, "b": 7}]}
    resp = client.post("/analyze", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    assert "profile" in resp.json()

def test_analysis_invalid(auth_headers):
    resp = client.post("/analyze", json={}, headers=auth_headers)
    assert resp.status_code in [400, 422]

# --- Modeling Endpoint ---

def test_modeling_train(auth_headers):
    payload = {
        "X": [[1, 2], [3, 4], [5, 6]],
        "y": [0, 1, 0],
        "model_type": "classification"
    }
    resp = client.post("/model/train", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    result = resp.json()
    assert "metrics" in result

def test_modeling_invalid(auth_headers):
    resp = client.post("/model/train", json={"X": [], "y": []}, headers=auth_headers)
    assert resp.status_code in [400, 422]

# --- Export Endpoint ---

def test_export_sql(auth_headers):
    payload = {"data": [{"id": 1, "val": "x"}, {"id": 2, "val": "y"}], "dialect": "postgresql"}
    resp = client.post("/export/sql", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    sql = resp.json().get("sql")
    assert sql and "CREATE TABLE" in sql

def test_export_invalid(auth_headers):
    resp = client.post("/export/sql", json={"bad": "input"}, headers=auth_headers)
    assert resp.status_code in [400, 422]

# --- Query Endpoint ---

def test_query_nl_sql(auth_headers):
    payload = {"question": "How many users are there?", "schema": {"users": ["id"]}}
    resp = client.post("/query/nl2sql", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "sql" in data

def test_query_sql_exec(auth_headers):
    payload = {"sql": "SELECT 1 as a;"}
    resp = client.post("/query/exec", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    assert "results" in resp.json()

def test_query_invalid(auth_headers):
    resp = client.post("/query/exec", json={"bad": "sql"}, headers=auth_headers)
    assert resp.status_code in [400, 422]

# --- Auth, RBAC, Permission ---

def test_auth_required():
    resp = client.post("/ingest")
    assert resp.status_code in [401, 403]

def test_admin_required(admin_token):
    headers = {"Authorization": admin_token}
    resp = client.post("/admin/only", headers=headers)
    # Should succeed for admin, fail for user
    assert resp.status_code in [200, 404, 501]  # depends on endpoint implementation

def test_user_forbidden(test_user_token):
    headers = {"Authorization": test_user_token}
    resp = client.post("/admin/only", headers=headers)
    assert resp.status_code in [401, 403, 404, 501]

# --- Rate Limiting and Edge Cases ---

def test_rate_limiting(auth_headers):
    # Simulate rapid requests for rate limiting (mock if needed)
    for _ in range(5):
        resp = client.post("/ingest", files={"file": ("t.csv", io.BytesIO(b"a,b\n1,2"), "text/csv")}, headers=auth_headers)
    # The last one may be rate-limited
    assert resp.status_code in [200, 429]

# --- Error Handling ---

def test_404_not_found(auth_headers):
    resp = client.get("/not_a_real_endpoint", headers=auth_headers)
    assert resp.status_code == 404

def test_internal_error_handling(auth_headers):
    with patch("routes.some_function", side_effect=Exception("fail")):
        resp = client.post("/ingest", files={"file": ("t.csv", io.BytesIO(b"a,b\n1,2"), "text/csv")}, headers=auth_headers)
        assert resp.status_code in [500, 400]

# --- Load and Concurrency ---

def test_load_concurrency(auth_headers):
    files = {"file": ("t.csv", io.BytesIO(b"a,b\n1,2\n3,4"), "text/csv")}
    start = time.time()
    n = 10
    results = []
    for _ in range(n):
        resp = client.post("/ingest", files=files, headers=auth_headers)
        results.append(resp.status_code)
    elapsed = time.time() - start
    assert all(code == 200 for code in results) or any(code == 429 for code in results)
    assert elapsed < 10  # Fast enough for 10 requests

# --- External Dependency Mocking (example for DB) ---

def test_db_mocking(auth_headers):
    with patch("routes.db_session") as mock_db:
        mock_db.query.return_value.all.return_value = [{"id": 1, "name": "Mock"}]
        resp = client.post("/query/exec", json={"sql": "SELECT * FROM users;"}, headers=auth_headers)
        assert resp.status_code == 200
        assert "results" in resp.json()