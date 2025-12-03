import os
from typing import List

# Load a single text file.
def load_text_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return ""

# Load multiple text files.
def load_multiple_files(file_paths: List[str]) -> List[str]:
    return [load_text_file(path) for path in file_paths]

# List all files in a directory with a specific extension.
def list_files_in_directory(directory: str, extension: str = "pdf") -> List[str]:
    try:
        files = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                # Check file extension
                if extension is None or filename.endswith(extension):
                    files.append(file_path)
        # Sort files naturally (handles numeric ordering correctly)
        files.sort()
        return files
    except Exception as e:
        print(f"Error listing directory {directory}: {e}")
        return []
