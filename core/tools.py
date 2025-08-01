"""File operation tools with logging support."""

import csv
import json
import os
import shutil
import logging
from typing import Dict, List, Optional, Union

from langchain.tools import tool
from langchain_community.tools import ShellTool

# Initialize logger
logger = logging.getLogger(__name__)


@tool
def list_files(directory_path: str) -> Union[Dict[str, List[str]], str]:
    """
    List files and subdirectories in the specified directory.

    Args:
        directory_path: Path to the directory

    Returns:
        Dictionary with 'files' and 'directories' keys containing lists of names,
        or error message string if operation fails

    Example:
        {
            "files": ["file1.txt", "file2.py"],
            "directories": ["subdir1", "subdir2"]
        }
    """
    logger.info(f"Listing directory contents: {directory_path}")
    directory_path = os.path.abspath(directory_path)
    try:
        if not os.path.isdir(directory_path):
            error_msg = f"Directory not found at '{directory_path}'"
            logger.error(error_msg)
            return f"Error: {error_msg}"

        contents = os.listdir(directory_path)
        result = {"files": [], "directories": []}

        for item in contents:
            full_path = os.path.join(directory_path, item)
            if os.path.isdir(full_path):
                result["directories"].append(item)
            else:
                result["files"].append(item)

        logger.debug(f"Directory listing result: {result}")
        return result
    except Exception as e:
        error_msg = f"Error listing directory '{directory_path}': {str(e)}"
        logger.exception(error_msg)
        return f"Error: {error_msg}"


@tool
def read_files(
    filepaths: List[str], preview: bool = False
) -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Read contents of multiple files.

    Args:
        filepaths: List of file paths to read
        preview: If True, return preview of file content (default: False)

    Returns:
        Dictionary mapping file paths to their contents or error messages

    Example:
        {
            "/path/file1.txt": "file content...",
            "/path/file2.txt": {"error": "File not found"}
        }
    """
    logger.info(f"Reading files: {filepaths}")
    results = {}

    for filepath in filepaths:
        filepath = os.path.abspath(filepath)
        try:
            if not os.path.exists(filepath):
                error_msg = f"File not found: '{filepath}'"
                logger.warning(error_msg)
                results[filepath] = {"error": error_msg}
                continue

            if not os.path.isfile(filepath):
                error_msg = f"Path is not a file: '{filepath}'"
                logger.warning(error_msg)
                results[filepath] = {"error": error_msg}
                continue

            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()
                if preview:
                    content = content[:1000] + "..." if len(content) > 1000 else content
                results[filepath] = content
                logger.debug(f"Successfully read file: {filepath}")

        except Exception as e:
            error_msg = f"Error reading file '{filepath}': {str(e)}"
            logger.exception(error_msg)
            results[filepath] = {"error": error_msg}

    return results


@tool
def preview_file_content(file_path: str) -> str:
    """
    Get preview of file content for CSV, JSON or TXT files.

    Args:
        file_path: Path to the file

    Returns:
        Formatted preview string with sample content and statistics

    Raises:
        Various file reading exceptions
    """
    logger.info(f"Generating preview for file: {file_path}")

    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        return f"Error: {error_msg}"

    if file_path.lower().endswith(".csv"):
        return _preview_csv(file_path)
    elif file_path.lower().endswith(".json"):
        return _preview_json(file_path)
    elif file_path.lower().endswith(".txt"):
        return _preview_text(file_path)
    else:
        error_msg = "Unsupported file type. Only CSV, JSON and TXT files are supported."
        logger.warning(error_msg)
        return error_msg


def _preview_csv(file_path: str) -> str:
    """Helper function to preview CSV files."""
    rows = []
    total_rows = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                total_rows += 1
                if total_rows <= 5:
                    rows.append(row)

        preview = ["CSV File Preview:"]
        preview.extend([", ".join(row) for row in rows])
        preview.append(f"Total rows: {total_rows}")
        return "\n".join(preview)
    except Exception as e:
        error_msg = f"Error reading CSV file: {e}"
        logger.exception(error_msg)
        return f"Error: {error_msg}"


def _preview_json(file_path: str) -> str:
    """Helper function to preview JSON files."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        preview = []
        if isinstance(data, dict):
            items = list(data.items())
            preview.append("JSON File Preview (first 5 key-value pairs):")
            for key, value in items[:5]:
                preview.append(f"{key}: {value}")
            preview.append(f"Total key-value pairs: {len(items)}")
        elif isinstance(data, list):
            preview.append("JSON File Preview (first 5 elements):")
            for item in data[:5]:
                preview.append(str(item))
            preview.append(f"Total elements: {len(data)}")
        else:
            return f"Unsupported JSON type: {type(data)}"

        return "\n".join(preview)
    except Exception as e:
        error_msg = f"Error reading JSON file: {e}"
        logger.exception(error_msg)
        return f"Error: {error_msg}"


def _preview_text(file_path: str) -> str:
    """Helper function to preview text files."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        words = content.split()
        preview = [
            "TXT File Preview (first 10000 words):",
            " ".join(words[:10000]),
            f"Total words: {len(words)}",
        ]
        return "\n".join(preview)
    except Exception as e:
        error_msg = f"Error reading TXT file: {e}"
        logger.exception(error_msg)
        return f"Error: {error_msg}"


def _tree(
    directory: str, level: Optional[int], prefix: str = "", current_depth: int = 0
) -> str:
    if not os.path.exists(directory):
        return f"Error: Directory not found: {directory}\n"

    if not os.path.isdir(directory):
        return f"Error: Path is not a directory: {directory}\n"

    result = []
    items = sorted(os.listdir(directory))
    is_last_level = (level is not None and current_depth + 1 == level)
    max_items = 3 if is_last_level else len(items)

    for index, item in enumerate(items[:max_items]):
        item_path = os.path.join(directory, item)
        is_last = index == min(len(items), max_items) - 1
        connector = "└── " if is_last else "├── "
        result.append(f"{prefix}{connector}{item}\n")

        if os.path.isdir(item_path) and not is_last_level:
            extension = "    " if is_last else "│   "
            result.append(
                _tree(item_path, level, prefix + extension, current_depth + 1)
            )

    # Add ellipsis if some items are not shown
    if is_last_level and len(items) > 3:
        result.append(f"{prefix}└── ...\n")

    return "".join(result)


@tool
def tree(directory: str, level: Optional[int] = 3) -> str:
    """
    Generate directory tree structure.

    Args:
        directory: Root directory path
        level: Maximum depth to display (None for unlimited)

    Returns:
        String representation of directory tree
    """
    return _tree(directory, level)


@tool
def create_directory(directory_path: str) -> str:
    """
    Create directory including parent directories if needed.

    Args:
        directory_path: Path of directory to create

    Returns:
        Success or error message
    """
    logger.info(f"Creating directory: {directory_path}")
    try:
        os.makedirs(directory_path, exist_ok=True)
        msg = f"Directory created or already exists at '{directory_path}'"
        logger.info(msg)
        return msg
    except Exception as e:
        error_msg = f"Error creating directory '{directory_path}': {str(e)}"
        logger.exception(error_msg)
        return f"Error: {error_msg}"


@tool
def write_to_file(path: str, content: str, mode: str = "w") -> str:
    """
    Write content to file, creating parent directories if needed.

    Args:
        path: File path
        content: Content to write
        mode: File open mode (default: 'w')

    Returns:
        Success or error message
    """
    logger.info(f"Writing to file: {path}")
    try:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created parent directories: {directory}")

        with open(path, mode) as file:
            file.write(content)

        msg = f"Successfully wrote to file '{path}'"
        logger.info(msg)
        return msg
    except Exception as e:
        error_msg = f"Error writing to file '{path}': {str(e)}"
        logger.exception(error_msg)
        return f"Error: {error_msg}"


@tool
def copy_file(src: str, dst: str) -> str:
    """
    Copy file from source to destination.

    Args:
        src: Source file path
        dst: Destination file path

    Returns:
        Success or error message
    """
    logger.info(f"Copying file from {src} to {dst}")
    try:
        shutil.copy2(src, dst)
        msg = f"Successfully copied file from '{src}' to '{dst}'"
        logger.info(msg)
        return msg
    except Exception as e:
        error_msg = f"Error copying file from '{src}' to '{dst}': {str(e)}"
        logger.exception(error_msg)
        return f"Error: {error_msg}"


@tool
def run_script(command: str) -> str:
    """
    Execute shell command using ShellTool.

    Args:
        command: Shell command to execute

    Returns:
        Command output or error message
    """
    logger.info(f"Executing command: {command}")
    try:
        shell_tool = ShellTool()
        result = shell_tool.run({"commands": [command]})
        logger.debug(f"Command execution result: {result}")
        return result
    except Exception as e:
        error_msg = f"Error executing command '{command}': {str(e)}"
        logger.exception(error_msg)
        return f"Error: {error_msg}"


if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.DEBUG)

    # Test each function using its invoke method
    import tempfile
    import shutil

    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp()
    try:
        print("Testing create_directory:")
        nested_dir = os.path.join(test_dir, "parent", "child")
        print(create_directory(nested_dir))

        print("\nTesting list_files:")
        test_file1 = os.path.join(test_dir, "test1.txt")
        test_file2 = os.path.join(test_dir, "test2.txt")
        with open(test_file1, "w") as f:
            f.write("Test content 1")
        with open(test_file2, "w") as f:
            f.write("Test content 2")
        print(list_files(test_dir))

        print("\nTesting read_files:")
        print(
            read_files(
                [test_file1, test_file2, os.path.join(test_dir, "nonexistent.txt")]
            )
        )

        print("\nTesting tree:")
        os.makedirs(os.path.join(test_dir, "dir1", "subdir1"))
        os.makedirs(os.path.join(test_dir, "dir2"))
        with open(os.path.join(test_dir, "dir1", "file1.txt"), "w") as f:
            f.write("File 1 content")
        with open(os.path.join(test_dir, "dir2", "file2.txt"), "w") as f:
            f.write("File 2 content")
        print(tree(test_dir))

        print("\nTesting write_to_file:")
        new_file = os.path.join(test_dir, "written.txt")
        print(write_to_file(new_file, "This is new content"))
        with open(new_file, "r") as f:
            print(f"Content verification: '{f.read()}'")

        print("\nTesting run_script:")
        print(run_script("echo 'Hello from shell'"))

    finally:
        shutil.rmtree(test_dir)
