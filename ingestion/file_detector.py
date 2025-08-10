import os
import mimetypes
import struct

# External libraries required:
# pip install pandas openpyxl xlrd pyarrow PyPDF2 lxml

import pandas as pd

from common.exceptions import UnsupportedFileTypeError, FileParsingError

from PyPDF2.errors import PdfReadError
from PyPDF2 import PdfReader
import pyarrow.parquet as pq

from lxml import etree

class UnsupportedFileType(Exception):
    pass

class CorruptedFile(Exception):
    pass

def _is_csv(filepath):
    try:
        with open(filepath, "rb") as f:
            sample = f.read(2048)
            sample_str = sample.decode(errors="replace")
            if "," in sample_str or ";" in sample_str or "\t" in sample_str:
                pd.read_csv(filepath, nrows=5)
                return True
    except Exception:
        return False
    return False

def _is_excel(filepath):
    try:
        if filepath.lower().endswith('.xlsx'):
            import openpyxl
            openpyxl.load_workbook(filepath, read_only=True)
            return True
        elif filepath.lower().endswith('.xls'):
            import xlrd
            xlrd.open_workbook(filepath, on_demand=True)
            return True
        pd.read_excel(filepath, nrows=1)
        return True
    except Exception:
        return False

def _is_json(filepath):
    try:
        with open(filepath, "rb") as f:
            start = f.read(2048).lstrip()
            if start.startswith(b'{') or start.startswith(b'['):
                pd.read_json(filepath, nrows=1)
                return True
    except Exception:
        return False
    return False

def _is_xml(filepath):
    try:
        with open(filepath, "rb") as f:
            start = f.read(1024).lstrip()
            if start.startswith(b'<'):
                from lxml import etree
                etree.parse(filepath)
                return True
    except Exception:
        return False
    return False
    
    
def _is_pdf(filepath):
    try:
        from PyPDF2.errors import PdfReadError
        from PyPDF2 import PdfReader
        with open(filepath, "rb") as f:
            reader = PdfReader(f)
            return len(reader.pages) > 0
    except Exception:
        return False
    return False

def _is_parquet(filepath):
    try:
        import pyarrow.parquet as pq
        pq.read_schema(filepath)
        return True
    except Exception:
        return False
    return False

def _is_sqldump(filepath):
    try:
        with open(filepath, "rb") as f:
            sample = f.read(4096)
            sample_str = sample.decode(errors="replace").lower()
            if ("create table" in sample_str or "insert into" in sample_str) and ("--" in sample_str or "/*" in sample_str):
                return True
            if sample_str.startswith("begin transaction") or sample_str.startswith("commit;"):
                return True
    except Exception:
        return False
    return False
    

def detect_file_type(filepath: str) -> str:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    mime, _ = mimetypes.guess_type(filepath)
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in [".csv"] or (mime and "csv" in mime):
            if _is_csv(filepath): return 'csv'
        if ext in [".xls", ".xlsx"] or (mime and "excel" in mime):
            if _is_excel(filepath): return 'excel'
        if ext in [".json"] or (mime and "json" in mime):
            if _is_json(filepath): return 'json'
        if ext in [".xml"] or (mime and "xml" in mime):
            if _is_xml(filepath): return 'xml'
        if ext in [".pdf"] or (mime and "pdf" in mime):
            if _is_pdf(filepath): return 'pdf'
        if ext in [".parquet"] or (mime and "parquet" in mime):
            if _is_parquet(filepath): return 'parquet'
        if ext in [".sql", ".dump"]:
            if _is_sqldump(filepath): return 'sqldump'
    except Exception as e:
        raise FileParsingError(filepath, None, cause=e)

    # Fallback: try by content
    try:
        if _is_csv(filepath): return 'csv'
        if _is_excel(filepath): return 'excel'
        if _is_json(filepath): return 'json'
        if _is_xml(filepath): return 'xml'
        if _is_pdf(filepath): return 'pdf'
        if _is_parquet(filepath): return 'parquet'
        if _is_sqldump(filepath): return 'sqldump'
    except Exception as e:
        raise FileParsingError(filepath, None, cause=e)

    raise UnsupportedFileTypeError(filepath, ext)

if __name__ == "__main__":
    import sys
    try:
        ftype = detect_file_type(sys.argv[1])
        print(f"Detected type: {ftype}")
    except Exception as e:
        print(f"Error: {e}")






