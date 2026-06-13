import os
import argparse
from collections import defaultdict

def analyze_folder(path):
    total_files = 0
    total_folders = 0
    total_size = 0
    file_types = defaultdict(int)

    for root, dirs, files in os.walk(path):
        total_folders += len(dirs)
        total_files += len(files)
        for file in files:
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                total_size += size
            except OSError:
                size = 0
            ext = os.path.splitext(file)[1].lower()
            file_types[ext] += 1

    # Report
    print(f"Analysis of folder: {path}")
    print(f"Total folders: {total_folders}")
    print(f"Total files: {total_files}")
    print(f"Total size: {total_size / (1024*1024):.2f} MB")
    print("File types breakdown:")
    for ext, count in file_types.items():
        ext_display = ext if ext else "[no extension]"
        print(f"  {ext_display}: {count} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a folder and its subfolders")
    parser.add_argument("folder", help="Path to the folder to analyze")
    args = parser.parse_args()

    if not os.path.exists(args.folder):
        print("The specified folder does not exist!")
    else:
        analyze_folder(args.folder)