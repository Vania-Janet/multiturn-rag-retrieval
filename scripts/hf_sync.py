#!/usr/bin/env python3
"""
Sync large files to Hugging Face Hub.

This script uploads large artifacts (models, indices, embeddings) that cannot
be stored in GitHub due to size limitations.

Usage:
    # Initial setup (authenticate)
    python scripts/hf_sync.py --setup
    
    # Upload everything
    python scripts/hf_sync.py --upload-all
    
    # Upload specific directories
    python scripts/hf_sync.py --upload artifacts/models
    python scripts/hf_sync.py --upload indices
    
    # Download artifacts from HF
    python scripts/hf_sync.py --download artifacts/models
    python scripts/hf_sync.py --download-all
    
    # Check what would be uploaded (dry run)
    python scripts/hf_sync.py --upload-all --dry-run
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
import subprocess

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will rely on env vars being set manually

try:
    from huggingface_hub import HfApi, login, snapshot_download, hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("ERROR: huggingface_hub not installed")
    print("Install with: pip install huggingface-hub")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Directories containing large files
LARGE_DIRS = [
    "artifacts/models",
    "artifacts/embeddings",
    "artifacts/logs",
    "src/pipeline/artifacts/models",
    "src/pipeline/artifacts/embeddings",
    "src/pipeline/artifacts/logs",
    "indices/clapnq",
    "indices/cloud",
    "indices/fiqa",
    "indices/govt",
    "data/processed",
]

# File patterns to include
LARGE_FILE_PATTERNS = [
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.bin",
    "*.index",
    "*.faiss",
    "*.pkl",
    "*.pickle",
    "*.h5",
    "*.ckpt",
    "*.jsonl",
    "*.parquet",
]

# Default HF repo format: username/rag-benchmark-artifacts
DEFAULT_REPO_ID = None  # Will be set from config or user input


def get_repo_id(config_path: Path = Path("configs/hf_sync.yaml")) -> str:
    """Get HF repo ID from config, env vars, or prompt user."""
    
    # 1. Check config file
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
            if config and "repo_id" in config:
                return config["repo_id"]
    
    # 2. Check environment variables
    username = os.getenv("HUGGINGFACE_USERNAME")
    repo_name = os.getenv("HUGGINGFACE_REPO")
    
    if username and repo_name:
        repo_id = f"{username}/{repo_name}"
        logger.info(f"Found repo_id in environment: {repo_id}")
        return repo_id

    # 3. Prompt user
    username = input("Enter your HuggingFace username: ").strip()
    repo_name = input("Enter repo name (default: rag-benchmark-artifacts): ").strip()
    if not repo_name:
        repo_name = "rag-benchmark-artifacts"
    
    repo_id = f"{username}/{repo_name}"
    
    # Save to config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump({"repo_id": repo_id}, f)
    
    logger.info(f"Saved repo_id to {config_path}")
    return repo_id


def setup_hf_auth():
    """Setup HuggingFace authentication."""
    logger.info("Setting up HuggingFace authentication...")
    
    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    
    if token:
        logger.info("Found HUGGINGFACE_TOKEN in environment")
        try:
            login(token=token)
            logger.info("✓ Authentication successful")
            return True
        except Exception as e:
            logger.error(f"Authentication with env token failed: {e}")
            # Fallback to interactive login
    
    logger.info("You'll need a HuggingFace write token from: https://huggingface.co/settings/tokens")
    
    try:
        login()
        logger.info("✓ Authentication successful")
        return True
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return False


def create_repo_if_not_exists(repo_id: str, private: bool = True):
    """Create HF repo if it doesn't exist."""
    api = HfApi()
    
    try:
        api.repo_info(repo_id=repo_id)
        logger.info(f"✓ Repo exists: https://huggingface.co/{repo_id}")
        return True
    except HfHubHTTPError:
        logger.info(f"Creating new repo: {repo_id}")
        try:
            api.create_repo(
                repo_id=repo_id,
                private=private,
                repo_type="model"  # Using model repo for storage
            )
            logger.info(f"✓ Created repo: https://huggingface.co/{repo_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create repo: {e}")
            return False


def get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    if path.is_file():
        return path.stat().st_size
    
    for item in path.rglob('*'):
        if item.is_file():
            total += item.stat().st_size
    return total


def format_size(bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"


def upload_directory(
    local_dir: Path,
    repo_id: str,
    path_in_repo: Optional[str] = None,
    dry_run: bool = False
) -> bool:
    """
    Upload a directory to HuggingFace Hub.
    
    Args:
        local_dir: Local directory to upload
        repo_id: HuggingFace repo ID
        path_in_repo: Path in repo (default: same as local)
        dry_run: If True, only show what would be uploaded
    """
    
    if not local_dir.exists():
        logger.warning(f"Directory does not exist: {local_dir}")
        return False
    
    # Calculate size
    size = get_directory_size(local_dir)
    if size == 0:
        logger.info(f"Skipping empty directory: {local_dir}")
        return True
    
    logger.info(f"Uploading {local_dir} ({format_size(size)})...")
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would upload to {repo_id}/{path_in_repo or local_dir}")
        return True
    
    try:
        api = HfApi()
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            path_in_repo=path_in_repo or str(local_dir),
            repo_type="model"
        )
        logger.info(f"✓ Uploaded {local_dir}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to upload {local_dir}: {e}")
        return False


def download_directory(
    repo_id: str,
    path_in_repo: str,
    local_dir: Path,
    force: bool = False
) -> bool:
    """
    Download a directory from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repo ID
        path_in_repo: Path in repo to download
        local_dir: Local directory to save to
        force: Overwrite existing files
    """
    
    if local_dir.exists() and not force:
        logger.warning(f"Directory exists (use --force to overwrite): {local_dir}")
        return False
    
    logger.info(f"Downloading {path_in_repo} from {repo_id}...")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(local_dir.parent),
            allow_patterns=[f"{path_in_repo}/**"]
        )
        logger.info(f"✓ Downloaded to {local_dir}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download {path_in_repo}: {e}")
        return False


def list_artifacts(verbose: bool = False):
    """List all large artifacts and their sizes."""
    logger.info("Scanning for large artifacts...")
    
    total_size = 0
    artifacts = []
    
    for dir_path in LARGE_DIRS:
        path = Path(dir_path)
        if path.exists():
            size = get_directory_size(path)
            total_size += size
            artifacts.append((path, size))
            
            if verbose:
                logger.info(f"  {dir_path}: {format_size(size)}")
    
    logger.info(f"\nTotal artifacts: {len(artifacts)}")
    logger.info(f"Total size: {format_size(total_size)}")
    
    return artifacts


def main():
    parser = argparse.ArgumentParser(
        description="Sync large files with HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Setup HuggingFace authentication"
    )
    
    parser.add_argument(
        "--upload",
        type=str,
        metavar="DIR",
        help="Upload specific directory"
    )
    
    parser.add_argument(
        "--upload-all",
        action="store_true",
        help="Upload all large artifacts"
    )
    
    parser.add_argument(
        "--download",
        type=str,
        metavar="DIR",
        help="Download specific directory"
    )
    
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Download all artifacts from HF"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all large artifacts"
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        help="HuggingFace repo ID (username/repo-name)"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Create private repo (default: True)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing files"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Setup
    if args.setup:
        success = setup_hf_auth()
        sys.exit(0 if success else 1)
    
    # List artifacts
    if args.list:
        list_artifacts(verbose=True)
        sys.exit(0)
    
    # Get repo ID
    repo_id = args.repo_id or get_repo_id()
    logger.info(f"Using repo: {repo_id}")
    
    # Create repo if needed
    if args.upload or args.upload_all:
        if not args.dry_run:
            create_repo_if_not_exists(repo_id, private=args.private)
    
    # Upload specific directory
    if args.upload:
        local_dir = Path(args.upload)
        success = upload_directory(local_dir, repo_id, dry_run=args.dry_run)
        sys.exit(0 if success else 1)
    
    # Upload all
    if args.upload_all:
        logger.info("Uploading all large artifacts...")
        artifacts = list_artifacts()
        
        failed = []
        for path, size in artifacts:
            if size == 0:
                continue
            success = upload_directory(path, repo_id, dry_run=args.dry_run)
            if not success:
                failed.append(path)
        
        if failed:
            logger.error(f"\n✗ Failed to upload {len(failed)} directories:")
            for path in failed:
                logger.error(f"  - {path}")
            sys.exit(1)
        else:
            logger.info("\n✓ All artifacts uploaded successfully!")
            logger.info(f"View at: https://huggingface.co/{repo_id}")
            sys.exit(0)
    
    # Download specific directory
    if args.download:
        path_in_repo = args.download
        local_dir = Path(path_in_repo)
        success = download_directory(repo_id, path_in_repo, local_dir, force=args.force)
        sys.exit(0 if success else 1)
    
    # Download all
    if args.download_all:
        logger.info("Downloading all artifacts from HF...")
        failed = []
        
        for dir_path in LARGE_DIRS:
            path = Path(dir_path)
            success = download_directory(repo_id, dir_path, path, force=args.force)
            if not success:
                failed.append(dir_path)
        
        if failed:
            logger.error(f"\n✗ Failed to download {len(failed)} directories:")
            for path in failed:
                logger.error(f"  - {path}")
            sys.exit(1)
        else:
            logger.info("\n✓ All artifacts downloaded successfully!")
            sys.exit(0)
    
    # No action specified
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
