#!/usr/bin/env python3

import os
from pathlib import Path

def debug_repository(repo_path):
    """Debug script to see what's in the repository."""
    repo_path = Path(repo_path)
    
    if not repo_path.exists():
        print(f"Repository path does not exist: {repo_path}")
        return
        
    if not repo_path.is_dir():
        print(f"Repository path is not a directory: {repo_path}")
        return
        
    print(f"Repository path: {repo_path}")
    print(f"Repository name: {repo_path.name}")
    
    # List all files
    all_files = list(repo_path.rglob('*'))
    print(f"Total files and directories: {len(all_files)}")
    
    # Separate files and directories
    files = [f for f in all_files if f.is_file()]
    dirs = [d for d in all_files if d.is_dir()]
    
    print(f"Files: {len(files)}")
    print(f"Directories: {len(dirs)}")
    
    # Show first 10 files
    print("\nFirst 10 files:")
    for i, file_path in enumerate(files[:10]):
        rel_path = file_path.relative_to(repo_path)
        print(f"  {i+1}. {rel_path}")
        
    # Check for .gitignore
    gitignore_path = repo_path / '.gitignore'
    if gitignore_path.exists():
        print(f"\n.gitignore exists with {len(gitignore_path.read_text().splitlines())} lines")
    else:
        print("\nNo .gitignore found")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        debug_repository(sys.argv[1])
    else:
        print("Usage: python debug_repo.py <repository_path>")