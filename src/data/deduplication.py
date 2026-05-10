#Might not be neccessary!

import subprocess
import os
from pathlib import Path
import numpy as np

class Deduplicator:
    def __init__(self, extracted_text_dir, output_dir, repo_path):
        self.extracted_text_dir = Path(extracted_text_dir)
        self.output_dir = Path(output_dir)
        self.repo_path = Path(repo_path)
        self.data_dir = self.output_dir / "data"
        self.cache_dir = self.output_dir / "cache"
        
        # Create necessary directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def prepare_combined_file(self):
        combined_file = self.data_dir / "combined_text.txt"
        size_file = self.data_dir / "combined_text.txt.size"
        
        # Gather all text files
        text_files = list(self.extracted_text_dir.glob('**/*.txt'))
        print(f"Combining {len(text_files)} text files")
        
        # Prepare size tracking
        sizes = [0]
        current_size = 0
        
        with open(combined_file, 'wb') as outfile:
            for file_path in text_files:
                with open(file_path, 'rb') as infile:
                    # Add a unique separator before each file
                    separator = b"\xff\xff" + file_path.stem.encode('utf-8') + b"\xff\xff"
                    outfile.write(separator)
                    
                    # Read and write the file content
                    content = infile.read()
                    outfile.write(content)
                    
                    # Update size tracking
                    current_size += len(separator) + len(content)
                    sizes.append(current_size)
        
        # Write size file (required by the deduplication code)
        with open(size_file, 'wb') as f:
            f.write(np.array(sizes, dtype=np.uint64).tobytes())
            
        return combined_file
    
    def run_deduplication_pipeline(self, length_threshold=100, num_threads=8):
        """Run the complete deduplication pipeline."""
        print("Starting deduplication pipeline...")
        
        # Step 1: Prepare combined text file
        combined_file = self.prepare_combined_file()
        print(f"Created combined file at {combined_file}")
        
        # Step 2: Create suffix array
        self._create_suffix_array(combined_file)
        
        # Step 3: Find duplicates
        self._find_duplicates(combined_file, length_threshold, num_threads)
        
        # Step 4: Collect duplicates
        byte_ranges_file = self._collect_duplicates(combined_file, length_threshold)
        
        # Step 5: Remove duplicates
        deduplicated_file = self._remove_duplicates(combined_file, byte_ranges_file)
        
        print(f"Deduplication complete. Deduplicated text saved to {deduplicated_file}")
        return deduplicated_file
    


    ########################################################################################################
    #https://github.com/google-research/deduplicate-text-datasets/blob/master/scripts/make_suffix_array.py
    ########################################################################################################

    def _create_suffix_array(self, text_file):
        print("Creating suffix array...")
        script_path = self.repo_path / "scripts" / "make_suffix_array.py"
        
        subprocess.run([
            "python3", 
            str(script_path), 
            str(text_file)
        ], check=True)
        
        return text_file.with_suffix('.txt.table.bin')
    
    def _find_duplicates(self, text_file, length_threshold, num_threads):
        """Find duplicate sequences in the text file."""
        print(f"Finding duplicates with length threshold {length_threshold}...")
        
        original_dir = os.getcwd()
        os.chdir(str(self.repo_path))
        
        subprocess.run([
            "cargo", "run", "self-similar",
            "--data-file", str(text_file),
            "--length-threshold", str(length_threshold),
            "--cache-dir", str(self.cache_dir),
            "--num-threads", str(num_threads)
        ], check=True)
        
        # Return to original directory
        os.chdir(original_dir)
    
    def _collect_duplicates(self, text_file, length_threshold):
        print("Collecting duplicates...")
        
        # Store current directory to return to it later
        original_dir = os.getcwd()
        os.chdir(str(self.repo_path))
        
        # Output file for byte ranges
        byte_ranges_file = self.output_dir / "duplicates.byterange"
        
        # Run the collect command
        with open(byte_ranges_file, 'w') as f:
            subprocess.run([
                "cargo", "run", "collect",
                "--data-file", str(text_file),
                "--cache-dir", str(self.cache_dir),
                "--length-threshold", str(length_threshold)
            ], stdout=f, check=True)
        
        # Return to original directory
        os.chdir(original_dir)
        
        return byte_ranges_file
    
    def _remove_duplicates(self, text_file, byte_ranges_file):
        print("Removing duplicates...")
        
        # Output file for deduplicated text
        deduplicated_file = self.output_dir / "deduplicated_text.txt"
        
        # Parse byte ranges
        byte_ranges = []
        with open(byte_ranges_file, 'r') as f:
            lines = f.readlines()
            found_start = False
            for line in lines:
                if 'out' in line:
                    found_start = True
                    continue
                if found_start:
                    try:
                        start, end = map(int, line.strip().split())
                        byte_ranges.append((start, end))
                    except ValueError:
                        continue
        
        # Remove duplicates by applying the byte ranges
        with open(text_file, 'rb') as infile, open(deduplicated_file, 'wb') as outfile:
            data = infile.read()
            
            # Sort ranges in reverse order to not affect indices when removing
            byte_ranges.sort(reverse=True)
            
            for start, end in byte_ranges:
                if start < len(data) and end <= len(data):
                    data = data[:start] + data[end:]
            
            outfile.write(data)
        
        # Also create a version that maps back to the original nodes
        self._map_deduplicated_to_nodes(deduplicated_file)
        
        return deduplicated_file
    
    def _map_deduplicated_to_nodes(self, deduplicated_file):
        print("Mapping deduplicated text to original nodes...")
        
        with open(deduplicated_file, 'rb') as f:
            deduplicated_data = f.read()
        
        # Split by the custom separator to get document boundaries
        separator_pattern = b"\xff\xff"
        parts = deduplicated_data.split(separator_pattern)
        
        # Reconstruct document structure
        documents = []
        current_doc = None
        
        for part in parts:
            if len(part) == 0:
                continue
                
            # Check if this could be a filename marker
            if separator_pattern not in part:
                if current_doc is not None:
                    current_doc["content"] += part
            else:
                # This might be a filename
                segments = part.split(separator_pattern)
                if segments and len(segments[0]) > 0:
                    # Complete previous document if exists
                    if current_doc is not None:
                        documents.append(current_doc)
                    
                    # Start new document
                    doc_id = segments[0].decode('utf-8', errors='ignore')
                    current_doc = {"id": doc_id, "content": b""}
        
        # Add the last document
        if current_doc is not None:
            documents.append(current_doc)
        
        # For each document, try to map content back to original nodes
        mapped_documents = []
        for doc in documents:
            doc_id = doc["id"]
            original_node_file = self.extracted_text_dir / f"{doc_id}.json"
            
            if original_node_file.exists():
                try:
                    with open(original_node_file, 'r', encoding='utf-8') as f:
                        original_doc = json.load(f)
                    
                    # Convert binary content to text
                    doc_content = doc["content"].decode('utf-8', errors='ignore')
                    
                    # Create a new document with deduplicated nodes
                    deduplicated_doc = {
                        "filename": original_doc["filename"],
                        "filepath": original_doc["filepath"],
                        "nodes": []
                    }
                    
                    # TODO: This is a simplistic approach - in a real implementation,
                    # you would want to map the deduplicated content back to nodes
                    # more intelligently using string matching or other techniques
                    
                    # For now, just create new nodes based on paragraphs
                    paragraphs = doc_content.split("\n\n")
                    for i, paragraph in enumerate(paragraphs):
                        if paragraph.strip():
                            deduplicated_doc["nodes"].append({
                                "text": paragraph,
                                "type": "TextNode",
                                "is_title": i == 0 and len(paragraph) < 200,  # Simple heuristic
                                "is_heading": False,
                                "metadata": {"source": "deduplicated"}
                            })
                    
                    mapped_documents.append(deduplicated_doc)
                    
                except Exception as e:
                    print(f"Error mapping document {doc_id}: {e}")
        
        # Save the mapped documents
        mapped_dir = self.output_dir / "mapped_documents"
        os.makedirs(mapped_dir, exist_ok=True)
        
        for doc in mapped_documents:
            doc_path = mapped_dir / f"{doc['filename'].split('.')[0]}_deduplicated.json"
            with open(doc_path, 'w', encoding='utf-8') as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(mapped_documents)} mapped documents to {mapped_dir}")