#!/usr/bin/env python3
"""
Extract text content from XML files for deduplication.
"""
import os
import sys
from pathlib import Path
import re
import xml.etree.ElementTree as ET

def extract_text_from_xml(xml_file):
    """Extract text content from XML file, removing tags."""
    try:
        with open(xml_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Try to parse as XML first to extract specific content
        try:
            # Parse the XML properly
            root = ET.fromstring(content)
            
            # Extract all text content recursively
            def get_all_text(element):
                text = element.text or ""
                for child in element:
                    text += get_all_text(child)
                text += element.tail or ""
                return text
            
            text = get_all_text(root)
        except ET.ParseError:
            # If XML parsing fails, fall back to regex
            text = re.sub(r'<[^>]+>', ' ', content)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    except Exception as e:
        print(f"Error processing {xml_file}: {e}")
        return ""

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 extract_xml_text.py <input_directory> <output_file>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Processing XML files from: {input_dir}")
    print(f"Output will be saved to: {output_file}")
    
    # Find all XML files
    xml_files = list(Path(input_dir).glob("*.xml"))
    print(f"Found {len(xml_files)} XML files")
    
    if not xml_files:
        print("No XML files found!")
        sys.exit(1)
    
    # Process each file and write to output
    total_bytes = 0
    doc_boundaries = [0]  # Track document boundaries for .size file
    processed_count = 0
    
    with open(output_file, 'wb') as out_f:  # Open in binary mode for accurate byte counting
        for i, xml_file in enumerate(xml_files):
            if i % 50 == 0:  # Progress update every 50 files
                print(f"Processing {i+1}/{len(xml_files)}: {xml_file.name}")
            
            text = extract_text_from_xml(xml_file)
            if text:
                # Convert to bytes and write
                text_bytes = (text + '\n\n').encode('utf-8')  # Document separator
                out_f.write(text_bytes)
                
                total_bytes += len(text_bytes)
                doc_boundaries.append(total_bytes)
                processed_count += 1
    
    # Create size file (tracks byte positions of document boundaries)
    size_file = output_file + '.size'
    with open(size_file, 'w') as size_f:
        for boundary in doc_boundaries[:-1]:  # Remove last boundary
            size_f.write(f"{boundary}\n")
    
    print(f"Extraction complete!")
    print(f"Total bytes: {total_bytes}")
    print(f"Documents processed: {processed_count}")
    print(f"Output file: {output_file}")
    print(f"Size file: {size_file}")

if __name__ == "__main__":
    main()
