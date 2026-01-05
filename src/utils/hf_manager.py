import os
import logging
from pathlib import Path
from typing import Optional, Union
from dotenv import load_dotenv

# Try to import huggingface_hub
try:
    from huggingface_hub import HfApi, login
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

logger = logging.getLogger(__name__)

class HFManager:
    """
    Manages interactions with Hugging Face Hub for uploading artifacts.
    """
    
    def __init__(self, repo_id: Optional[str] = None):
        load_dotenv()
        self.token = os.getenv("HF_TOKEN")
        self.repo_id = repo_id or os.getenv("HF_REPO_ID")
        self.api = HfApi() if HF_AVAILABLE else None
        self.enabled = False
        
        if not HF_AVAILABLE:
            logger.debug("huggingface_hub not installed. Uploads will be skipped.")
            return

        if not self.token:
            logger.debug("HF_TOKEN not found in environment. Uploads will be skipped.")
            return
            
        if not self.repo_id:
            logger.debug("HF_REPO_ID not found. Uploads will be skipped.")
            return

        try:
            login(token=self.token)
            self.enabled = True
            logger.info(f"✅ Hugging Face integration enabled. Target: {self.repo_id}")
        except Exception as e:
            logger.error(f"Failed to login to Hugging Face: {e}")
            self.api = None

    def upload_file(self, file_path: Union[str, Path], path_in_repo: Optional[str] = None):
        """Upload a single file to the repository."""
        if not self.enabled:
            return

        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return

        if path_in_repo is None:
            # Preserve relative path structure if possible, else use filename
            # Assuming we are running from project root
            try:
                path_in_repo = str(file_path.relative_to(Path.cwd()))
            except ValueError:
                path_in_repo = file_path.name

        try:
            logger.info(f"🚀 Uploading {file_path.name} to Hugging Face...")
            self.api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type="dataset"
            )
            logger.info(f"✅ Successfully uploaded {file_path.name}")
        except Exception as e:
            logger.error(f"❌ Failed to upload {file_path}: {e}")

    def upload_directory(self, dir_path: Union[str, Path], path_in_repo: Optional[str] = None):
        """Upload a directory to the repository."""
        if not self.enabled:
            return

        dir_path = Path(dir_path)
        if not dir_path.exists():
            logger.error(f"Directory not found: {dir_path}")
            return

        if path_in_repo is None:
            try:
                path_in_repo = str(dir_path.relative_to(Path.cwd()))
            except ValueError:
                path_in_repo = dir_path.name

        try:
            logger.info(f"🚀 Uploading directory {dir_path.name} to Hugging Face...")
            self.api.upload_folder(
                folder_path=dir_path,
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type="dataset"
            )
            logger.info(f"✅ Successfully uploaded directory {dir_path.name}")
        except Exception as e:
            logger.error(f"❌ Failed to upload directory {dir_path}: {e}")
