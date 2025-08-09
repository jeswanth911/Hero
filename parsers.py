import os
import logging
import pandas as pd

try:
    import camelot
except ImportError:
    camelot = None

try:
    import tabula
except ImportError:
    tabula = None

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

import json
import xml.etree.ElementTree as ET
import sqlite3

# Setup logger
logger = logging.getLogger("parsers")
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def parse_csv(filepath, encoding="utf-8") -> pd.DataFrame:
    """Parse a CSV file into a DataFrame, handling encoding issues."""
    try:
        logger.info(f"Parsing CSV file: {filepath}")
        try:
            df = pd.read_csv(filepath, encoding=encoding)
        except UnicodeDecodeError:
            logger.warning("UTF-8 decoding failed, trying 'latin1' encoding...")
            df = pd.read_csv(filepath, encoding="latin1")
        return df
    except Exception as e:
        logger.error(f"Failed to parse CSV: {e}")
        raise


def parse_excel(filepath, sheet_name=0) -> pd.DataFrame:
    """Parse Excel file(s) into a DataFrame. If multiple sheets, returns first sheet."""
    try:
        logger.info(f"Parsing Excel file: {filepath}")
        excel = pd.ExcelFile(filepath)
        logger.info(f"Available sheets: {excel.sheet_names}")
        if isinstance(sheet_name, int) or isinstance(sheet_name, str):
            df = pd.read_excel(filepath, sheet_name=sheet_name)
        else:
            # Return all sheets as dict of DataFrames
            df = pd.read_excel(filepath, sheet_name=None)
        return df
    except Exception as e:
        logger.error(f"Failed to parse Excel: {e}")
        raise


def parse_json(filepath) -> pd.DataFrame:
    """Parse JSON file into a DataFrame."""
    try:
        logger.info(f"Parsing JSON file: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Try different structures
        if isinstance(data, dict):
            # If dict of lists, or dict of dicts
            if all(isinstance(v, list) for v in data.values()):
                # Try to concatenate all lists
                df = pd.concat([pd.DataFrame(v) for v in data.values()], ignore_index=True)
            else:
                df = pd.json_normalize(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            logger.error("Unexpected JSON structure.")
            raise ValueError("Unexpected JSON structure.")
        return df
    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}")
        raise


def parse_xml(filepath, record_tag=None) -> pd.DataFrame:
    """Parse XML file into a DataFrame."""
    try:
        logger.info(f"Parsing XML file: {filepath}")
        tree = ET.parse(filepath)
        root = tree.getroot()
        # Try to guess record tag if not provided
        if record_tag is None:
            if len(root) == 0:
                raise ValueError("No records found in XML.")
            record_tag = root[0].tag
        records = []
        for elem in root.findall(f".//{record_tag}"):
            record = {}
            for child in elem:
                record[child.tag] = child.text
            records.append(record)
        df = pd.DataFrame(records)
        return df
    except Exception as e:
        logger.error(f"Failed to parse XML: {e}")
        raise


def parse_pdf_table(filepath, flavor="stream", backend="camelot", pages="all") -> pd.DataFrame:
    """
    Parse tables from a PDF file into a DataFrame.
    Tries Camelot or Tabula.
    """
    try:
        logger.info(f"Parsing PDF tables from: {filepath} using {backend}")
        if backend == "camelot":
            if camelot is None:
                raise ImportError("Camelot is not installed.")
            tables = camelot.read_pdf(filepath, flavor=flavor, pages=pages)
            if len(tables) == 0:
                raise ValueError("No tables found in PDF.")
            # Return the first table, or concatenate all
            df = pd.concat([t.df for t in tables], ignore_index=True)
        elif backend == "tabula":
            if tabula is None:
                raise ImportError("Tabula is not installed.")
            dfs = tabula.read_pdf(filepath, pages=pages, multiple_tables=True)
            if len(dfs) == 0:
                raise ValueError("No tables found in PDF.")
            df = pd.concat(dfs, ignore_index=True)
        else:
            raise ValueError("Unknown PDF backend. Use 'camelot' or 'tabula'.")
        return df
    except Exception as e:
        logger.error(f"Failed to parse PDF: {e}")
        raise


def parse_parquet(filepath) -> pd.DataFrame:
    """Parse Parquet file into a DataFrame."""
    try:
        logger.info(f"Parsing Parquet file: {filepath}")
        if pq is None:
            raise ImportError("pyarrow is not installed.")
        df = pd.read_parquet(filepath)
        return df
    except Exception as e:
        logger.error(f"Failed to parse Parquet: {e}")
        raise


def parse_sql_dump(filepath) -> pd.DataFrame:
    """
    Parse a SQL dump file into a DataFrame.
    Only supports dumps that can be imported into SQLite.
    Returns the first table found.
    """
    try:
        logger.info(f"Parsing SQL dump: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            sql_script = f.read()
        # Create in-memory SQLite DB
        conn = sqlite3.connect(":memory:")
        try:
            conn.executescript(sql_script)
            # Get first table name
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            tables = cursor.fetchall()
            if not tables:
                raise ValueError("No tables found in SQL dump.")
            table_name = tables[0][0]
            logger.info(f"Extracting table: {table_name}")
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            return df
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Failed to parse SQL dump: {e}")
        raise


def parse_file(filepath: str, file_type: str, **kwargs) -> pd.DataFrame:
    """
    Dispatch function to parse a file based on its type.
    file_type: One of 'csv', 'excel', 'json', 'xml', 'pdf', 'parquet', 'sql'
    """
    logger.info(f"Dispatching parsing for type '{file_type}' from: {filepath}")
    file_type = file_type.lower()
    if file_type == "csv":
        return parse_csv(filepath, **kwargs)
    elif file_type == "excel":
        return parse_excel(filepath, **kwargs)
    elif file_type == "json":
        return parse_json(filepath)
    elif file_type == "xml":
        return parse_xml(filepath, **kwargs)
    elif file_type == "pdf":
        return parse_pdf_table(filepath, **kwargs)
    elif file_type == "parquet":
        return parse_parquet(filepath)
    elif file_type == "sql":
        return parse_sql_dump(filepath)
    else:
        logger.error(f"Unsupported file type: {file_type}")
        raise ValueError(f"Unsupported file type: {file_type}")


# If run as script, demonstrate dispatcher
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python parsers.py <filepath> <filetype>")
        sys.exit(1)
    filepath = sys.argv[1]
    file_type = sys.argv[2]
    df = parse_file(filepath, file_type)
    print(df.head())