#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import subprocess
import shutil
import re

def extract_folder_id(url_or_id):
    if not url_or_id:
        return None
    
    if url_or_id.startswith('http'):
        match = re.search(r'/folders/([a-zA-Z0-9_-]+)', url_or_id)
        if match:
            return match.group(1)
        return None
    return url_or_id

def check_rclone():
    try:
        result = subprocess.run(['rclone', 'version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def get_rclone_remote_name():
    try:
        result = subprocess.run(['rclone', 'listremotes'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            remotes = result.stdout.strip().split('\n')
            for remote in remotes:
                if 'drive' in remote.lower() or 'gdrive' in remote.lower():
                    return remote.strip().rstrip(':')
            return remotes[0].strip().rstrip(':')
        return None
    except Exception:
        return None

def download_with_rclone_by_path(folder_path, output_dir, remote_name='gdrive'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not folder_path.endswith('/'):
        folder_path = folder_path + '/'
    
    print(f"Attempting to download with rclone from: {remote_name}:{folder_path}")
    
    try:
        result = subprocess.run(
            ['rclone', 'copy', f'{remote_name}:{folder_path}', str(output_dir), '-P'],
            capture_output=False,
            text=True
        )
        if result.returncode == 0:
            downloaded_files = list(output_dir.rglob("*"))
            xml_files = list(output_dir.rglob("*.xml"))
            if downloaded_files:
                print(f"\n✓ Download complete with rclone! Downloaded {len(downloaded_files)} files ({len(xml_files)} XML files)")
                print(f"Files saved to: {output_dir}")
                return True
        else:
            print(f"✗ Rclone error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error with rclone: {e}")
        return False

def download_with_rclone_by_id(folder_id, output_dir):
    remote_name = get_rclone_remote_name()
    if not remote_name:
        print("✗ No rclone remote found")
        return False
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Attempting to download with rclone using folder ID: {folder_id}")
    print("Note: rclone may need the folder path instead of ID. Trying alternative methods...")
    
    try:
        result = subprocess.run(
            ['rclone', 'lsf', f'{remote_name}:', '--dirs-only', '--drive-shared-with-me'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("\nAvailable folders in your Google Drive root:")
            folders = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
            for i, folder in enumerate(folders[:20], 1):
                print(f"  {i}. {folder}")
            if len(folders) > 20:
                print(f"  ... and {len(folders) - 20} more")
            
            print(f"\nTo download, you need to find the folder path. Try:")
            print(f"  python list_gdrive_folders.py")
            print(f"  python list_gdrive_folders.py 'folder_name'")
            print(f"\nThen use:")
            print(f"  python download_and_process_gdrive.py 'found/path/to/folder' {output_dir} --rclone-path")
            return False
        else:
            print(f"Could not list folders: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def install_gdown():
    try:
        import gdown
        return True
    except ImportError:
        print("Installing gdown...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import gdown
            return True
        except Exception:
            print("Failed to install gdown. Please install manually: pip install gdown")
            return False

def download_with_gdown(folder_id, output_dir):
    import gdown
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading files from Google Drive folder ID: {folder_id}")
    print(f"Output directory: {output_dir}")
    
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    
    try:
        print("Note: If the folder is not publicly accessible, you may need to:")
        print("  1. Make the folder publicly accessible (Viewer permission), OR")
        print("  2. Use gdown with authentication")
        
        gdown.download_folder(url, output=str(output_dir), quiet=False, use_cookies=False)
        
        downloaded_files = list(output_dir.rglob("*"))
        xml_files = list(output_dir.rglob("*.xml"))
        
        if not downloaded_files:
            print("⚠️  No files were downloaded. The folder might be private or require authentication.")
            return False
        
        print(f"\n✓ Download complete! Downloaded {len(downloaded_files)} files ({len(xml_files)} XML files)")
        print(f"Files saved to: {output_dir}")
        return True
    except Exception as e:
        error_msg = str(e)
        if "more than 50 files" in error_msg.lower():
            print(f"⚠️  gdown limitation: {error_msg}")
            print("Switching to rclone (no file limit)...")
            return None
        print(f"✗ Error downloading with gdown: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the folder is publicly accessible (Viewer permission)")
        print("2. Or use gdown with authentication")
        return False

def download_folder_from_gdrive(folder_id_or_path, output_dir, use_rclone_path=False):
    folder_id = extract_folder_id(folder_id_or_path)
    
    if use_rclone_path:
        remote_name = get_rclone_remote_name()
        if remote_name:
            if download_with_rclone_by_path(folder_id_or_path, output_dir, remote_name):
                return True
        else:
            print("⚠️  No rclone remote found. Trying gdown instead...")
    
    if folder_id:
        if install_gdown():
            gdown_result = download_with_gdown(folder_id, output_dir)
            if gdown_result is True:
                return True
            elif gdown_result is None:
                print("\n" + "="*60)
                print("gdown hit 50+ file limit. Switching to rclone...")
                print("="*60)
                if check_rclone():
                    remote_name = get_rclone_remote_name()
                    if remote_name:
                        print("\nTo use rclone, you need to find the folder path in your Google Drive.")
                        print("Let me help you find it...")
                        if download_with_rclone_by_id(folder_id, output_dir):
                            return True
                        print("\nPlease find the folder path manually:")
                        print("1. Run: python list_gdrive_folders.py")
                        print("2. Navigate to find your folder")
                        print("3. Then run: python download_and_process_gdrive.py 'path/to/folder' output_dir --rclone-path")
                        return False
                    else:
                        print("✗ No rclone remote configured")
                else:
                    print("✗ rclone not installed")
    else:
        print("✗ Invalid folder ID or path")
        print("Expected format: folder ID (e.g., 1-6zpe7G96lZ7y2jdlvXwlaKtv6s0AFAC)")
        print("Or full URL: https://drive.google.com/drive/folders/FOLDER_ID")
        print("Or rclone path: path/to/folder (use --rclone-path flag)")
        return False
    
    print("\n✗ Download failed.")
    print("\nAlternative options:")
    print("1. Find the folder path and use rclone:")
    print("   python list_gdrive_folders.py")
    print("   python download_and_process_gdrive.py 'path/to/folder' output_dir --rclone-path")
    print("2. Manually download files from Google Drive and place them in:")
    print(f"   {output_dir}")
    print(f"   Then run: python extract_xml_text.py {output_dir}")
    return False

def process_xml_files(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    input_dir = Path(input_dir)
    xml_files = list(input_dir.rglob("*.xml"))
    
    if not xml_files:
        print(f"No XML files found in {input_dir}")
        return
    
    print(f"\nFound {len(xml_files)} XML files to process")
    
    sys.path.insert(0, str(Path(__file__).parent))
    from extract_xml_text import process_xml_file
    
    success_count = 0
    for i, xml_file in enumerate(xml_files, 1):
        if process_xml_file(xml_file, output_dir):
            success_count += 1
        if i % 10 == 0:
            print(f"Progress: {i}/{len(xml_files)} files processed...")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count}/{len(xml_files)} XML files")
    print(f"Text files saved to: {output_dir}")
    print(f"{'='*60}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python download_and_process_gdrive.py <folder_id_or_path> [output_dir] [--rclone-path]")
        print("\nExamples:")
        print("  # Using Google Drive folder ID (uses gdown):")
        print("  python download_and_process_gdrive.py 1-6zpe7G96lZ7y2jdlvXwlaKtv6s0AFAC seed/ClimateRiskManagement")
        print("\n  # Using rclone path (if you know the folder path in your Drive):")
        print("  python download_and_process_gdrive.py 'ClimateRiskManagement' seed/ClimateRiskManagement --rclone-path")
        print("\n  # Using full Google Drive URL:")
        print("  python download_and_process_gdrive.py 'https://drive.google.com/drive/folders/1-6zpe7G96lZ7y2jdlvXwlaKtv6s0AFAC' seed/ClimateRiskManagement")
        sys.exit(1)
    
    folder_id_or_path = sys.argv[1]
    use_rclone_path = '--rclone-path' in sys.argv
    
    if len(sys.argv) > 2 and not sys.argv[2].startswith('--'):
        output_dir = Path(sys.argv[2])
    else:
        output_dir = Path("seed/ClimateRiskManagement")
    
    temp_download_dir = output_dir / "downloaded_files"
    
    print(f"{'='*60}")
    print("Step 1: Downloading files from Google Drive")
    print(f"{'='*60}")
    
    if not download_folder_from_gdrive(folder_id_or_path, temp_download_dir, use_rclone_path=use_rclone_path):
        print("\n⚠️  Download failed. If you already have the XML files, you can process them directly:")
        print(f"   python extract_xml_text.py {output_dir}")
        response = input("\nDo you want to continue with processing existing files? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
        temp_download_dir = output_dir
    
    print(f"\n{'='*60}")
    print("Step 2: Processing XML files to text")
    print(f"{'='*60}")
    
    process_xml_files(temp_download_dir, output_dir)
    
    print(f"\n{'='*60}")
    print("Step 3: Cleaning up downloaded files")
    print(f"{'='*60}")
    
    if temp_download_dir.exists() and temp_download_dir != output_dir and temp_download_dir.name == "downloaded_files":
        response = input(f"Remove temporary download directory '{temp_download_dir}'? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(temp_download_dir)
            print(f"✓ Removed temporary download directory: {temp_download_dir}")
        else:
            print(f"⚠️  Keeping temporary directory: {temp_download_dir}")
    
    print(f"\n🎉 All done! Text files are in: {output_dir}")

if __name__ == "__main__":
    main()

