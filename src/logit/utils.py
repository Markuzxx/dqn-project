import os


def write_to_file(file_path: str,
                  content: str) -> None:
    
    mode = 'a' if os.path.exists(file_path) else 'w'
    with open(file_path, mode) as f:
        f.write(content + '\n')
