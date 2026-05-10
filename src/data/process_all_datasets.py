#!/usr/bin/env python3
import subprocess
import os
import sys

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    print(f"✅ Success: {description}")
    return True

def process_dataset(folder_name, p_number):
    """Process a single dataset"""
    print(f"\n{'='*60}")
    print(f"🚀 PROCESSING P{p_number}: {folder_name}")
    print(f"{'='*60}")
    
    # Step 1: Download from Google Drive
    if not run_command(
        f'rclone copy "climateBERT:Climate_Articles/{folder_name}" "{folder_name}"',
        f"Downloading {folder_name} from Google Drive"
    ):
        return False
    
    # Step 2: Extract text from XML files
    if not run_command(
        f'python3 extract_xml_text.py "{folder_name}" processed_data/P{p_number}_original.txt',
        f"Extracting text from {folder_name}"
    ):
        return False
    
    # Step 3: Build suffix array
    os.chdir("deduplicate-text-datasets")
    if not run_command(
        f'python3 scripts/make_suffix_array.py ../processed_data/P{p_number}_original.txt',
        f"Building suffix array for P{p_number}"
    ):
        os.chdir("..")
        return False
    
    # Step 4: Find duplicates
    if not run_command(
        f'cargo run self-similar --data-file ../processed_data/P{p_number}_original.txt --length-threshold 100 --cache-dir /tmp/P{p_number}_cache --num-threads 4',
        f"Finding duplicates in P{p_number}"
    ):
        os.chdir("..")
        return False
    
    # Step 5: Collect duplicate ranges
    if not run_command(
        f'cargo run collect --data-file ../processed_data/P{p_number}_original.txt --cache-dir /tmp/P{p_number}_cache --length-threshold 100 > /tmp/P{p_number}.remove.byterange',
        f"Collecting duplicate ranges for P{p_number}"
    ):
        os.chdir("..")
        return False
    
    os.chdir("..")
    
    # Step 6: Clean the ranges file
    if not run_command(
        f'tail -n +7 /tmp/P{p_number}.remove.byterange > /tmp/P{p_number}.remove.byterange.clean',
        f"Cleaning ranges file for P{p_number}"
    ):
        return False
    
    # Step 7: Create and run deduplication script
    dedup_script = f"""#!/usr/bin/env python3

input_file = "processed_data/P{p_number}_original.txt"
ranges_file = "/tmp/P{p_number}.remove.byterange.clean"
output_file = "processed_data/P{p_number}_deduplicated.txt"

# Read ranges to remove
ranges_to_remove = []
try:
    with open(ranges_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('S') and not line.startswith('Merging'):
                parts = line.split()
                if len(parts) == 2:
                    try:
                        start, end = map(int, parts)
                        ranges_to_remove.append((start, end))
                    except ValueError:
                        continue
except FileNotFoundError:
    print(f"No ranges file found for P{p_number} - no duplicates to remove")
    # Just copy the original file
    import shutil
    shutil.copy(input_file, output_file)
    print(f"P{p_number}: No deduplication needed")
    exit(0)

ranges_to_remove.sort()
print(f"P{p_number}: Found {{len(ranges_to_remove)}} ranges to remove")

# Read original file
with open(input_file, 'rb') as f:
    data = f.read()

print(f"P{p_number}: Original file size: {{len(data):,}} bytes")

# Create ranges to keep
keep_ranges = []
current_pos = 0

for start, end in ranges_to_remove:
    if current_pos < start:
        keep_ranges.append((current_pos, start))
    current_pos = max(current_pos, end)

if current_pos < len(data):
    keep_ranges.append((current_pos, len(data)))

# Write deduplicated data
total_kept = 0
with open(output_file, 'wb') as f:
    for start, end in keep_ranges:
        chunk = data[start:end]
        f.write(chunk)
        total_kept += len(chunk)

print(f"P{p_number}: Deduplicated file size: {{total_kept:,}} bytes")
print(f"P{p_number}: Removed: {{len(data) - total_kept:,}} bytes ({{100 * (len(data) - total_kept) / len(data):.2f}}%)")
print(f"P{p_number}: SUCCESS! Clean dataset ready")
"""
    
    with open(f"remove_duplicates_P{p_number}.py", "w") as f:
        f.write(dedup_script)
    
    if not run_command(
        f'python3 remove_duplicates_P{p_number}.py',
        f"Deduplicating P{p_number}"
    ):
        return False
    
    # Cleanup
    run_command(f'rm -rf "{folder_name}"', f"Cleaning up downloaded folder")
    run_command(f'rm remove_duplicates_P{p_number}.py', f"Cleaning up script")
    
    return True

if __name__ == "__main__":
    # Get list of folders from Google Drive
    result = subprocess.run('rclone lsd "climateBERT:Climate_Articles"', shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Failed to list Google Drive folders")
        sys.exit(1)
    
    folders = []
    for line in result.stdout.strip().split('\n'):
        if line.strip():
            # Extract folder name from rclone lsd output
            parts = line.strip().split()
            if len(parts) >= 5:
                folder_name = ' '.join(parts[4:])
                folders.append(folder_name)
    
    print(f"📁 Found {len(folders)} datasets to process:")
    for i, folder in enumerate(folders, 1):
        print(f"  P{i}: {folder}")
    
    # Process each dataset
    successful = 0
    for i, folder in enumerate(folders, 1):
        if process_dataset(folder, i):
            successful += 1
        else:
            print(f"❌ Failed to process P{i}: {folder}")
    
    print(f"\n🎉 PROCESSING COMPLETE!")
    print(f"✅ Successfully processed: {successful}/{len(folders)} datasets")
    print(f"📁 Check processed_data/ for your P1, P2, ... P{len(folders)} files")
