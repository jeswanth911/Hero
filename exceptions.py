class IngestionError(Exception):
    """
    Base exception for all ingestion-related errors.
    """
    def __init__(self, message: str, filename: str = None, filetype: str = None, cause: Exception = None):
        self.filename = filename
        self.filetype = filetype
        self.cause = cause
        details = f"{message}"
        if filename:
            details += f" | File: {filename}"
        if filetype:
            details += f" | Filetype: {filetype}"
        if cause:
            details += f" | Cause: {repr(cause)}"
        super().__init__(details)


class UnsupportedFileTypeError(IngestionError):
    """
    Raised when an unsupported file type is encountered.
    """
    def __init__(self, filename: str, filetype: str, cause: Exception = None):
        message = f"Unsupported file type: {filetype}"
        super().__init__(message, filename=filename, filetype=filetype, cause=cause)


class FileParsingError(IngestionError):
    """
    Raised when a file cannot be parsed into a DataFrame.
    """
    def __init__(self, filename: str, filetype: str = None, cause: Exception = None):
        message = f"Failed to parse file"
        super().__init__(message, filename=filename, filetype=filetype, cause=cause)


class BatchProcessingError(IngestionError):
    """
    Raised for errors during batch processing of files.
    """
    def __init__(self, filename: str = None, filetype: str = None, cause: Exception = None):
        message = f"Batch processing error"
        super().__init__(message, filename=filename, filetype=filetype, cause=cause)