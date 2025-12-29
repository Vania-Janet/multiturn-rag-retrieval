#!/usr/bin/env python3
"""
Quick helper to check file sizes and recommend storage location.

Usage:
    python scripts/check_file_sizes.py
    python scripts/check_file_sizes.py --dir artifacts/models
    python scripts/check_file_sizes.py --threshold 50  # MB
"""

import argparse
from pathlib import Path
import os


def get_size(path: Path) -> int:
    """Get size of file or directory in bytes."""
    if path.is_file():
        return path.stat().st_size
    
    total = 0
    for item in path.rglob('*'):
        if item.is_file():
            try:
                total += item.stat().st_size
            except (OSError, PermissionError):
                pass
    return total


def format_size(bytes: int) -> str:
    """Format bytes as human-readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"


def get_storage_recommendation(size_bytes: int, path: Path) -> tuple[str, str]:
    """
    Recommend storage location based on size and file type.
    
    Returns (storage, emoji)
    """
    size_mb = size_bytes / (1024 * 1024)
    
    # Check if it's code/config
    if path.suffix in ['.py', '.yaml', '.yml', '.md', '.txt', '.json', '.toml']:
        if size_mb < 10:
            return ("GitHub", "✅")
        else:
            return ("GitHub (consider splitting)", "⚠️")
    
    # Check if it's a large model/data file
    large_extensions = ['.pt', '.pth', '.safetensors', '.bin', '.faiss', '.index', 
                       '.pkl', '.pickle', '.h5', '.ckpt', '.parquet']
    if path.suffix in large_extensions:
        return ("HuggingFace", "💾")
    
    # Size-based recommendations
    if size_mb < 1:
        return ("GitHub", "✅")
    elif size_mb < 10:
        return ("GitHub", "✅")
    elif size_mb < 100:
        return ("GitHub (Git LFS) or HuggingFace", "⚠️")
    else:
        return ("HuggingFace", "💾")


def check_directory(directory: Path, threshold_mb: float = 100, recursive: bool = True):
    """Check all files in directory."""
    
    print(f"\n{'='*80}")
    print(f"Scanning: {directory}")
    print(f"Threshold: {threshold_mb} MB")
    print(f"{'='*80}\n")
    
    files_info = []
    
    if recursive:
        iterator = directory.rglob('*')
    else:
        iterator = directory.glob('*')
    
    for item in iterator:
        if item.is_file():
            try:
                size = get_size(item)
                size_mb = size / (1024 * 1024)
                
                if size_mb >= threshold_mb:
                    storage, emoji = get_storage_recommendation(size, item)
                    rel_path = item.relative_to(directory.parent)
                    files_info.append((rel_path, size, storage, emoji))
            except (OSError, PermissionError):
                pass
    
    # Sort by size descending
    files_info.sort(key=lambda x: x[1], reverse=True)
    
    if not files_info:
        print(f"✓ No files larger than {threshold_mb} MB found")
        return
    
    print(f"{'File':<50} {'Size':<12} {'Storage':<30} {'Status'}")
    print(f"{'-'*50} {'-'*12} {'-'*30} {'-'*6}")
    
    total_size = 0
    for path, size, storage, emoji in files_info:
        total_size += size
        path_str = str(path)[:47] + "..." if len(str(path)) > 50 else str(path)
        print(f"{path_str:<50} {format_size(size):<12} {storage:<30} {emoji}")
    
    print(f"\n{'='*80}")
    print(f"Total: {len(files_info)} files, {format_size(total_size)}")
    print(f"{'='*80}\n")
    
    # Summary by storage type
    github_files = [f for f in files_info if "GitHub" in f[2]]
    hf_files = [f for f in files_info if "HuggingFace" in f[2]]
    
    if github_files:
        github_size = sum(f[1] for f in github_files)
        print(f"✅ GitHub: {len(github_files)} files, {format_size(github_size)}")
    
    if hf_files:
        hf_size = sum(f[1] for f in hf_files)
        print(f"💾 HuggingFace: {len(hf_files)} files, {format_size(hf_size)}")


def check_all_large_dirs():
    """Check all directories that might have large files."""
    
    large_dirs = [
        "artifacts/models",
        "artifacts/embeddings",
        "artifacts/logs",
        "indices",
        "data/processed",
        "data/raw",
    ]
    
    print("\n" + "="*80)
    print("CHECKING ALL LARGE DIRECTORIES")
    print("="*80)
    
    total_size = 0
    for dir_name in large_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            size = get_size(dir_path)
            total_size += size
            print(f"\n{dir_name:<30} {format_size(size):>12}")
    
    print(f"\n{'='*80}")
    print(f"TOTAL SIZE: {format_size(total_size)}")
    print(f"{'='*80}")
    
    if total_size > 100 * 1024 * 1024:  # > 100 MB
        print(f"\n💾 Recommendation: Upload to HuggingFace")
        print(f"   Run: python scripts/hf_sync.py --upload-all")
    else:
        print(f"\n✅ Total size is manageable")


def main():
    parser = argparse.ArgumentParser(description="Check file sizes and recommend storage")
    parser.add_argument(
        "--dir",
        type=Path,
        help="Directory to check"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10,
        help="Size threshold in MB (default: 10)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all large directories"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't recurse into subdirectories"
    )
    
    args = parser.parse_args()
    
    if args.all:
        check_all_large_dirs()
    elif args.dir:
        check_directory(args.dir, args.threshold, not args.no_recursive)
    else:
        # Default: check current directory
        check_directory(Path.cwd(), args.threshold, not args.no_recursive)


if __name__ == "__main__":
    main()
