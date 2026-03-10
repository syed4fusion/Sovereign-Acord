import os
import sys

# --- Configuration ---
# You can customize which file extensions to include in the count.
# Files with extensions NOT in this list will be skipped.
# To count ALL files, set this to an empty set: IGNORED_EXTENSIONS = set()
IGNORED_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico',  # Images
    '.mp4', '.mov', '.avi', '.mkv',                   # Videos
    '.mp3', '.wav', '.flac',                          # Audio
    '.zip', '.tar', '.gz', '.rar',                    # Archives
    '.exe', '.dll', '.so', '.o', '.pyc',              # Binaries/Compiled
    '.log', '.bin', '.pdf', '.docx', '.xlsx', '.db', '.pyc', '.jsonl', '.json', '.md', '.txt'   # Documents/Other data
}
# ---------------------


def count_lines_in_file(filepath):
    """
    Opens a file and counts the number of lines.
    Includes error handling for binary files.
    """
    try:
        # 'r' mode opens for reading (text mode)
        # We use a context manager ('with open...') to ensure the file is closed.
        with open(filepath, 'r', encoding='utf-8') as f:
            # A generator expression and sum() is the fastest way to count lines
            # in a file without loading the entire file into memory.
            return sum(1 for _ in f)
    except UnicodeDecodeError:
        # This error typically means the file is a binary file (e.g., an image, a video, a compiled file).
        # We print a message and return 0 lines for this file.
        print(f"  [SKIPPING] Binary or non-text file: {filepath}")
        return 0
    except Exception as e:
        # Catch other possible file errors (e.g., permission denied)
        print(f"  [ERROR] Could not read file {filepath}: {e}")
        return 0

def count_lines_in_directory(start_path):
    """
    Traverses the directory recursively and counts the total lines of code.
    """
    total_lines = 0
    total_files = 0
    
    # os.walk generates the file names in a directory tree
    # by walking the tree either top-down or bottom-up.
    for root, _, files in os.walk(start_path):
        for filename in files:
            # Get the full path to the file
            filepath = os.path.join(root, filename)
            
            # Check file extension against the ignored set
            _, ext = os.path.splitext(filename)
            if ext.lower() in IGNORED_EXTENSIONS:
                print(f"  [SKIPPING] Ignored extension: {filepath}")
                continue

            # Count the lines
            line_count = count_lines_in_file(filepath)
            
            if line_count > 0:
                total_lines += line_count
                total_files += 1
                # Optional: print line count for each file
                # print(f"  {line_count:<5} lines in {filepath}")
            
    return total_lines, total_files

if __name__ == "__main__":
    # 1. Determine the path to crawl
    if len(sys.argv) > 1:
        # Use the path provided as a command-line argument
        folder_to_crawl = sys.argv[1]
    else:
        # If no argument is provided, use the current directory
        folder_to_crawl = os.getcwd()

    print(f"--- Starting Line of Code (LOC) Counter ---")
    print(f"Target directory: {folder_to_crawl}")
    
    if not os.path.isdir(folder_to_crawl):
        print(f"\n[FATAL ERROR] The path '{folder_to_crawl}' is not a valid directory.")
        sys.exit(1)

    # 2. Run the main counting function
    final_loc, final_files = count_lines_in_directory(folder_to_crawl)

    # 3. Print the final result
    print("-" * 40)
    print(f"| RESULTS:")
    print(f"| Total files counted: {final_files}")
    print(f"| **Total Lines of Code:** {final_loc}")
    print("-" * 40)