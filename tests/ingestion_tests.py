import io
import json
import time
import tempfile
import pytest
import pandas as pd

from typing import Any, Dict
from fastapi.testclient import TestClient

# Assuming the API is in routes.py as `app`
from routes import app

client = TestClient(app)

# ====== Fixtures ======

@pytest.fixture(scope="module")
def sample_csv():
    return "a,b,c\n1,2,3\n4,5,6"

@pytest.fixture(scope="module")
def sample_excel(tmp_path_factory):
    import openpyxl
    tmp_dir = tmp_path_factory.mktemp("data")
    file_path = tmp_dir / "sample.xlsx"
    df = pd.DataFrame({"a": [1, 4], "b": [2, 5], "c": [3, 6]})
    df.to_excel(file_path, index=False)
    return str(file_path)

@pytest.fixture(scope="module")
def sample_json():
    return json.dumps([{"a": 1, "b": 2}, {"a": 4, "b": 5}])

@pytest.fixture(scope="module")
def sample_xml():
    return """<root>
        <row><a>1</a><b>2</b></row>
        <row><a>4</a><b>5</b></row>
    </root>"""

@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory):
    # Minimal PDF with tabular content (simulate, don't parse for real)
    file_path = tmp_path_factory.mktemp("data") / "sample.pdf"
    with open(file_path, "wb") as f:
        f.write(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")
    return str(file_path)

@pytest.fixture(scope="module")
def sample_db_dump(tmp_path_factory):
    file_path = tmp_path_factory.mktemp("data") / "sample.sql"
    with open(file_path, "w") as f:
        f.write("CREATE TABLE test (a INT, b INT); INSERT INTO test VALUES (1,2),(3,4);")
    return str(file_path)

@pytest.fixture(scope="module")
def malformed_csv():
    return "a,b\n1,2,3\nx"

@pytest.fixture(scope="module")
def large_csv(tmp_path_factory):
    # Generate a large CSV (100,000 rows)
    file_path = tmp_path_factory.mktemp("data") / "large.csv"
    df = pd.DataFrame({"a": range(100_000), "b": range(100_000)})
    df.to_csv(file_path, index=False)
    return str(file_path)

# ====== Parametrized File Type Detection Test ======

@pytest.mark.parametrize(
    "filename,content,expected_type",
    [
        ("data.csv", "a,b\n1,2", "csv"),
        ("table.xlsx", None, "excel"),
        ("data.json", '[{"a":1}]', "json"),
        ("data.xml", "<root></root>", "xml"),
        ("file.pdf", None, "pdf"),
        ("dump.sql", "CREATE TABLE test;", "sql"),
        ("badfile.txt", "randomtext", "unknown"),
    ]
)
def test_file_type_detection(filename, content, expected_type, sample_excel, sample_pdf, sample_db_dump):
    # Here, implement your file type detection logic or call your function
    # Example placeholder:
    def detect_type(filename, content=None):
        if filename.endswith(".csv"):
            return "csv"
        if filename.endswith(".xlsx"):
            return "excel"
        if filename.endswith(".json"):
            return "json"
        if filename.endswith(".xml"):
            return "xml"
        if filename.endswith(".pdf"):
            return "pdf"
        if filename.endswith(".sql"):
            return "sql"
        return "unknown"

    if filename.endswith(".xlsx"):
        assert detect_type(filename, None) == expected_type
    elif filename.endswith(".pdf"):
        assert detect_type(filename, None) == expected_type
    else:
        assert detect_type(filename, content) == expected_type

# ====== Parsing Correctness and Error Handling ======

@pytest.mark.parametrize(
    "fixture,filename,content_type,expect_success",
    [
        ("sample_csv", "file.csv", "text/csv", True),
        ("sample_excel", "file.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", True),
        ("sample_json", "file.json", "application/json", True),
        ("malformed_csv", "bad.csv", "text/csv", False),
    ]
)
def test_file_parsing_and_error_handling(fixture, filename, content_type, expect_success, request):
    file_content = request.getfixturevalue(fixture)
    if filename.endswith(".xlsx"):
        with open(file_content, "rb") as f:
            data = f.read()
    else:
        data = file_content.encode() if isinstance(file_content, str) else file_content

    files = {"file": (filename, io.BytesIO(data), content_type)}
    resp = client.post("/ingest", files=files, auth=("admin", "adminpw"))
    if expect_success:
        assert resp.status_code == 200
        body = resp.json()
        assert "columns" in body and "preview" in body
    else:
        assert resp.status_code == 400 or resp.status_code == 422

# ====== Batch and Streaming Ingestion with Large Datasets ======

def test_batch_and_streaming_large_csv(large_csv):
    with open(large_csv, "rb") as f:
        data = f.read()
    files = {"file": ("large.csv", io.BytesIO(data), "text/csv")}
    start = time.time()
    resp = client.post("/ingest", files=files, auth=("admin", "adminpw"))
    elapsed = time.time() - start
    assert resp.status_code == 200
    # Should not take "too long" (arbitrary: <10s for test)
    assert elapsed < 10

# ====== Mocked API Uploads and Edge Case Handling ======

def test_upload_empty_file():
    files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    resp = client.post("/ingest", files=files, auth=("admin", "adminpw"))
    assert resp.status_code == 400

def test_upload_unsupported_type():
    files = {"file": ("unknown.xyz", io.BytesIO(b"data"), "application/octet-stream")}
    resp = client.post("/ingest", files=files, auth=("admin", "adminpw"))
    assert resp.status_code in [400, 415]

# ====== Exception Verification ======

def test_ingestion_exception_handling(monkeypatch):
    # Simulate a failure in parsing
    def fake_parser(file):
        raise ValueError("Parse error")
    from routes import parse_uploaded_file
    monkeypatch.setattr("routes.parse_uploaded_file", fake_parser)
    files = {"file": ("fail.csv", io.BytesIO(b"a,b\n1,2"), "text/csv")}
    resp = client.post("/ingest", files=files, auth=("admin", "adminpw"))
    assert resp.status_code == 400
    assert "File parsing error" in resp.json().get("detail", "")

# ====== Performance Benchmark ======

@pytest.mark.benchmark(group="ingestion")
def test_ingest_performance_benchmark(large_csv, benchmark):
    with open(large_csv, "rb") as f:
        data = f.read()
    files = {"file": ("large.csv", io.BytesIO(data), "text/csv")}
    def do_ingest():
        resp = client.post("/ingest", files=files, auth=("admin", "adminpw"))
        assert resp.status_code == 200
    benchmark(do_ingest)
