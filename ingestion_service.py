import os
from typing import List, Dict, Any, Optional, Callable, Union
import pandas as pd
import logging

import file_detector  # Should provide detect_file_type(filepath) -> str
import parsers        # Should provide parse_file(filepath, file_type) -> pd.DataFrame
import batch_handler  # Should provide process_batch(filepaths: List[str]) -> List[pd.DataFrame]

logger = logging.getLogger("ingestion_service")
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

class IngestionService:
    """
    Orchestrates the full ingestion workflow: file detection, parsing, batch handling, and error capture.
    Suitable for use in API layers or backend services.
    """

    def __init__(
        self,
        log_hook: Optional[Callable[[str, str], None]] = None,
        progress_hook: Optional[Callable[[int, int, str], None]] = None,
        error_hook: Optional[Callable[[str, Exception], None]] = None,
    ):
        """
        log_hook(level:str, message:str): Called for important log messages.
        progress_hook(current:int, total:int, filename:str): For progress reporting.
        error_hook(filename:str, exception:Exception): Called on errors per file.
        """
        self.log_hook = log_hook
        self.progress_hook = progress_hook
        self.error_hook = error_hook

    def log(self, level: str, message: str):
        logger.log(getattr(logging, level.upper(), logging.INFO), message)
        if self.log_hook:
            self.log_hook(level, message)

    def handle_error(self, filename: str, error: Exception):
        self.log("error", f"Error with {filename}: {error}")
        if self.error_hook:
            self.error_hook(filename, error)

    def ingest_files(
        self,
        filepaths: List[str],
        parallel: bool = False,
        return_errors: bool = True,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, Any]]]:
        """
        Ingest a batch of files (paths only). Returns mapping from filename to DataFrame or error dict.
        """
        self.log("info", f"Starting ingestion of {len(filepaths)} files. Parallel: {parallel}")
        results = batch_handler.process_batch(
            filepaths,
            parallel=parallel,
            progress_hook=self.progress_hook,
            **kwargs
        )
        out: Dict[str, Union[pd.DataFrame, Dict[str, Any]]] = {}
        for idx, path in enumerate(filepaths):
            fname = os.path.basename(path)
            df = results[idx]
            if df is not None:
                out[fname] = df
            else:
                error_info = {
                    "error": f"Failed to process {fname}",
                }
                if return_errors:
                    out[fname] = error_info
                self.handle_error(fname, Exception(error_info["error"]))
        self.log("info", "Ingestion workflow complete.")
        return out

    def ingest_file(
        self,
        filepath: str,
        return_error: bool = True,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, Any]]]:
        """
        Ingest a single file (path only). Returns mapping from filename to DataFrame or error dict.
        """
        fname = os.path.basename(filepath)
        try:
            self.log("info", f"Detecting file type for {fname}")
            file_type = file_detector.detect_file_type(filepath)
            self.log("info", f"Detected file type '{file_type}' for {fname}")
            df = parsers.parse_file(filepath, file_type, **kwargs)
            self.log("info", f"Successfully parsed {fname}")
            return {fname: df}
        except Exception as e:
            self.handle_error(fname, e)
            if return_error:
                return {fname: {"error": str(e)}}
            else:
                return {}

    def ingest_uploads(
        self,
        uploads: List[Any],
        upload_handler: Callable[[Any], str],
        parallel: bool = False,
        return_errors: bool = True,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, Any]]]:
        """
        Accepts file uploads (e.g., Werkzeug/FileStorage, Django UploadedFile, etc.).
        The upload_handler should take an upload object and return a temp file path.
        """
        self.log("info", f"Received {len(uploads)} uploads.")
        filepaths = []
        for upload in uploads:
            try:
                path = upload_handler(upload)
                filepaths.append(path)
                self.log("info", f"Uploaded file saved as {path}")
            except Exception as e:
                fname = getattr(upload, 'filename', 'unknown')
                self.handle_error(fname, e)
                if return_errors:
                    return {fname: {"error": str(e)}}
        return self.ingest_files(filepaths, parallel=parallel, return_errors=return_errors, **kwargs)

    # For API: generic entrypoint
    def ingest(
        self,
        sources: List[Union[str, Any]],
        uploads: bool = False,
        upload_handler: Optional[Callable[[Any], str]] = None,
        parallel: bool = False,
        return_errors: bool = True,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, Any]]]:
        """
        Ingests either filepaths (if uploads=False) or upload objects (uploads=True).
        """
        if uploads:
            if upload_handler is None:
                raise ValueError("upload_handler must be provided for upload ingestion.")
            return self.ingest_uploads(
                uploads=sources,
                upload_handler=upload_handler,
                parallel=parallel,
                return_errors=return_errors,
                **kwargs
            )
        else:
            return self.ingest_files(
                filepaths=sources,
                parallel=parallel,
                return_errors=return_errors,
                **kwargs
            )

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingestion_service.py <file1> <file2> ... [--parallel]")
        sys.exit(1)
    filepaths = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    parallel = "--parallel" in sys.argv
    svc = IngestionService()
    results = svc.ingest(filepaths, parallel=parallel)
    for fname, result in results.items():
        if isinstance(result, pd.DataFrame):
            print(f"\nFile: {fname}\n{result.head()}\n")
        else:
            print(f"\nFile: {fname}\nError: {result['error']}\n")