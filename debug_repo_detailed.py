#!/usr/bin/env python3

import os
import sys
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
    print(f"\nTotal files and directories: {len(all_files)}")
    
    # Separate files and directories
    files = [f for f in all_files if f.is_file()]
    dirs = [d for d in all_files if d.is_dir()]
    
    print(f"Files: {len(files)}")
    print(f"Directories: {len(dirs)}")
    
    # Show first 20 files with extensions
    print(f"\nFirst 20 files:")
    source_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.cc', '.cxx', '.c', '.h', '.hpp', 
                        '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.clj', '.hs', '.ml', '.fs', 
                        '.lua', '.sh', '.bash', '.zsh', '.sql'}
    
    source_files = []
    for file_path in files:
        if file_path.suffix.lower() in source_extensions:
            source_files.append(file_path)
    
    print(f"Source files found: {len(source_files)}")
    
    for i, file_path in enumerate(source_files[:20]):
        rel_path = file_path.relative_to(repo_path)
        size = file_path.stat().st_size
        print(f"  {i+1:2d}. {rel_path} ({size} bytes)")
        
    # Check for .gitignore
    gitignore_path = repo_path / '.gitignore'
    if gitignore_path.exists():
        print(f"\n.gitignore exists with {len(gitignore_path.read_text().splitlines())} lines")
        print("First 10 lines of .gitignore:")
        for i, line in enumerate(gitignore_path.read_text().splitlines()[:10]):
            print(f"  {i+1:2d}. {line}")
    else:
        print("\nNo .gitignore found")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_repository(sys.argv[1])
    else:
        print("Usage: python debug_repo.py <repository_path>")