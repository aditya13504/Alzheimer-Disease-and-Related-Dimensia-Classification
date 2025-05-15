import os
import hashlib
from collections import defaultdict

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

def compute_md5(file_path, chunk_size=65536):
    """Compute MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def find_and_remove_duplicates(root_dirs):
    # Step 1: Group files by size
    size_map = defaultdict(list)
    for root_dir in root_dirs:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        size = os.path.getsize(file_path)
                        size_map[size].append(file_path)
                    except Exception:
                        continue
    # Step 2: For files with same size, check hash
    hash_map = defaultdict(list)
    for file_list in size_map.values():
        if len(file_list) < 2:
            continue
        for file_path in file_list:
            try:
                file_hash = compute_md5(file_path)
                hash_map[file_hash].append(file_path)
            except Exception:
                continue
    # Step 3: Remove duplicates (keep the first occurrence)
    removed = 0
    for file_list in hash_map.values():
        for duplicate in file_list[1:]:
            try:
                os.remove(duplicate)
                print(f"Removed duplicate: {duplicate}")
                removed += 1
            except Exception:
                print(f"Error removing {duplicate}")
    print(f"Total duplicates removed: {removed}")

if __name__ == "__main__":
    train_dir = os.path.join(os.getcwd(), 'train')
    test_dir = os.path.join(os.getcwd(), 'test')
    find_and_remove_duplicates([train_dir, test_dir])
