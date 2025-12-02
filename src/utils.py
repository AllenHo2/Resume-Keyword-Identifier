import os
from typing import List

def load_text_file(file_path: str) -> str:
    """
    Load text content from a file.
    
    Args:
        file_path: Path to the text file
    
    Returns:
        File content as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return ""

def load_multiple_files(file_paths: List[str]) -> List[str]:
    """
    Load multiple text files.
    
    Args:
        file_paths: List of file paths
    
    Returns:
        List of file contents
    """
    return [load_text_file(path) for path in file_paths]

def save_text_file(content: str, file_path: str):
    """
    Save text content to a file.
    
    Args:
        content: Text content to save
        file_path: Destination file path
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Saved to {file_path}")
    except Exception as e:
        print(f"Error saving file {file_path}: {e}")

def file_exists(file_path: str) -> bool:
    """
    Check if a file exists.
    
    Args:
        file_path: Path to check
    
    Returns:
        True if file exists, False otherwise
    """
    return os.path.exists(file_path)

def list_files_in_directory(directory: str, extension: str = None) -> List[str]:
    """
    List all files in a directory, optionally filtered by extension.
    
    Args:
        directory: Directory path
        extension: File extension to filter (e.g., '.txt', '.pdf')
    
    Returns:
        List of file paths
    """
    try:
        files = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                if extension is None or filename.endswith(extension):
                    files.append(file_path)
        return files
    except Exception as e:
        print(f"Error listing directory {directory}: {e}")
        return []
