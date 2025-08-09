import os
import logging
from typing import List, Callable, Optional, Any
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Assume these modules exist and are in the Python path
import parsers
import file_detector  # Should provide detect_file_type(filepath) -> str

# Setup logger
logger = logging.getLogger("batch_handler")
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

def default_progress(current: int, total: int, filename: str):
    logger.info(f"Processing file {current}/{total}: {filename}")

def stream_csv(filepath, chunk_size=100_000, encoding="utf-8") -> pd.DataFrame:
    """Stream and concatenate large CSV in chunks."""
    logger.info(f"Streaming CSV file: {filepath} in chunks of {chunk_size}")
    try:
        chunks = pd.read_csv(filepath, chunksize=chunk_size, encoding=encoding)
        df = pd.concat(chunks, ignore_index=True)
        return df
    except Exception as e:
        logger.warning(f"UTF-8 decoding failed for {filepath}, trying 'latin1'")
        try:
            chunks = pd.read_csv(filepath, chunksize=chunk_size, encoding="latin1")
            df = pd.concat(chunks, ignore_index=True)
            return df
        except Exception as ex:
            logger.error(f"Failed to stream CSV: {ex}")
            raise

def stream_parquet(filepath, row_group_size: Optional[int] = None) -> pd.DataFrame:
    """Stream parquet file in row groups (requires pyarrow)."""
    logger.info(f"Streaming Parquet file: {filepath}")
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(filepath)
        df = table.to_pandas()
        return df
    except Exception as e:
        logger.error(f"Failed to stream Parquet: {e}")
        raise

def stream_large_file(filepath, file_type: str) -> pd.DataFrame:
    """Dispatch streaming for large files based on file type."""
    if file_type == "csv":
        return stream_csv(filepath)
    elif file_type == "parquet":
        return stream_parquet(filepath)
    else:
        # For other types, fallback to regular parser
        logger.warning(f"No streaming implemented for file type: {file_type}, using standard parser.")
        return parsers.parse_file(filepath, file_type)

def process_file(filepath: str, 
                 progress_hook: Optional[Callable[[int, int, str], None]] = None, 
                 index: int = 1, 
                 total: int = 1) -> Optional[pd.DataFrame]:
    """Process a single file: detect type, parse, handle large files."""
    filename = os.path.basename(filepath)
    try:
        if progress_hook:
            progress_hook(index, total, filename)
        logger.info(f"Detecting file type for {filename}")
        file_type = file_detector.detect_file_type(filepath)
        # Check file size (e.g., >100MB)
        size = os.path.getsize(filepath)
        if file_type in ("csv", "parquet") and size > 100 * 1024 * 1024:
            logger.info(f"{filename} is large ({size/1024/1024:.2f} MB). Using chunk-wise streaming.")
            df = stream_large_file(filepath, file_type)
        else:
            df = parsers.parse_file(filepath, file_type)
        logger.info(f"Successfully processed {filename}")
        return df
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        return None

def process_batch(
    filepaths: List[str],
    parallel: bool = False,
    progress_hook: Optional[Callable[[int, int, str], None]] = default_progress,
    max_workers: int = 4,
) -> List[Optional[pd.DataFrame]]:
    """
    Process a batch of files, returning a list of DataFrames.
    Continues processing if any file fails.
    Supports sequential or parallel processing.
    """
    results = [None] * len(filepaths)
    logger.info(f"Starting batch processing of {len(filepaths)} files. Parallel: {parallel}")
    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_file, path, progress_hook, idx + 1, len(filepaths)): idx
                for idx, path in enumerate(filepaths)
            }
            for i, future in enumerate(as_completed(future_to_idx)):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Exception in parallel processing for file {filepaths[idx]}: {e}")
    else:
        for idx, path in enumerate(filepaths):
            results[idx] = process_file(path, progress_hook, idx + 1, len(filepaths))
    logger.info("Batch processing completed.")
    return results

# Example usage in script mode
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python batch_handler.py <file1> <file2> ... [--parallel]")
        sys.exit(1)
    filepaths = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    parallel = "--parallel" in sys.argv
    dfs = process_batch(filepaths, parallel=parallel)
    for idx, df in enumerate(dfs):
        if df is not None:
            print(f"\nFile {idx+1}: {filepaths[idx]}\n{df.head()}\n")
        else:
            print(f"\nFile {idx+1}: {filepaths[idx]} FAILED to process.\n")