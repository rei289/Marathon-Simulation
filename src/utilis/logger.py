"""Script to set up logging for the StrideSim project."""
import logging
import sys
from pathlib import Path

from google.cloud import storage


class StrideSimLogger:
    """A logger class for the StrideSim project that logs to both console and a file."""

    def __init__(self,
                 execution_env: str,
                 bucket_name: str|None,
                 folder_name: str,
                 job_id: str,
                 log_file:str="console.log") -> None:
        """Initialize the logger with a name, folder, and log file name."""
        self.execution_env = execution_env
        self.name = "StrideSim"
        self.folder_name = folder_name
        self.job_id = job_id
        self.log_file = log_file

        # Use /tmp on GCP containers because it is writable.
        if execution_env == "gcp":
            self.local_root = "/tmp/stride_sim" #noqa S108
        else:
            self.local_root = bucket_name or "logs"

        self.log_dir = Path(self.local_root) / self.folder_name / self.job_id
        self.log_path = self.log_dir / self.log_file

    def setup_logger(self) -> logging.Logger:
        """Set up a logger that writes to both console and a file."""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # prevent log messages from being propagated to the root logger

        # Prevent duplicate logs if setup_logger is called more than once.
        if logger.handlers:
            for handler in logger.handlers:
                handler.close()
            logger.handlers.clear()

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create formatters and add them to the handlers
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Create handlers
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(self.log_dir / self.log_file)
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    def close_logger(self, logger: logging.Logger) -> None:
        """Flush and close all handlers."""
        for handler in logger.handlers[:]:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)

    def upload_log_to_gcs(
        self,
        bucket_name: str,
        destination_blob: str | None = None,
    ) -> str:
        """Upload the local log file to GCS and return the blob path."""
        if not self.log_path.exists():
            error = f"Log file not found: {self.log_path}"
            raise FileNotFoundError(error)

        blob_path = (
            destination_blob
            if destination_blob is not None
            else f"{self.folder_name}/{self.job_id}/{self.log_file}"
        )

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(self.log_path))
        return blob_path
