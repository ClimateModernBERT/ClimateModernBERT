#!/usr/bin/env python3
"""
Extract clean text content from Elsevier XML files.
Removes all XML tags, metadata, and noise, keeping only readable text.
"""
import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import re
from typing import List, Optional

# Namespaces used in Elsevier XML files
NAMESPACES = {
    'ce': 'http://www.elsevier.com/xml/common/dtd',
    'ja': 'http://www.elsevier.com/xml/ja/dtd',
    'xocs': 'http://www.elsevier.com/xml/xocs/dtd',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'prism': 'http://prismstandard.org/namespaces/basic/2.0/',
    'xlink': 'http://www.w3.org/1999/xlink',
    'mml': 'http://www.w3.org/1998/Math/MathML',
    'cals': 'http://www.elsevier.com/xml/common/cals/dtd',
    'sb': 'http://www.elsevier.com/xml/common/struct-bib/dtd',
    'sa': 'http://www.elsevier.com/xml/common/struct-aff/dtd',
    'tb': 'http://www.elsevier.com/xml/common/table/dtd',
}

def extract_text_from_element(element: ET.Element, text_parts: List[str]) -> None:
    """
    Recursively extract text from XML elements, handling various Elsevier XML structures.
    """
    # Get direct text content
    if element.text and element.text.strip():
        text_parts.append(element.text.strip())
    
    # Process child elements
    for child in element:
        # Skip certain elements that contain metadata/noise
        tag = child.tag
        if any(skip in tag for skip in ['xocs:', 'ref-info', 'attachment', 'link', 'object']):
            continue
        
        # Extract text from child
        extract_text_from_element(child, text_parts)
        
        # Get tail text (text after the element)
        if child.tail and child.tail.strip():
            text_parts.append(child.tail.strip())

def extract_article_text(xml_file: Path) -> str:
    """
    Extract clean text content from an Elsevier XML file.
    """
    try:
        # Parse XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        text_parts = []
        
        # Find the article element (main content)
        # Try different possible locations
        article = None
        
        # Method 1: Look for article in ja namespace
        article = root.find('.//{http://www.elsevier.com/xml/ja/dtd}article')
        
        # Method 2: Look for article without namespace
        if article is None:
            for elem in root.iter():
                if 'article' in elem.tag.lower() and elem.tag.endswith('article'):
                    article = elem
                    break
        
        # Method 3: Look for sections directly
        if article is None:
            sections = root.findall('.//{http://www.elsevier.com/xml/common/dtd}sections')
            if sections:
                article = root
        
        # If we found an article element, extract from it
        if article is not None:
            # Extract title
            title = article.find('.//{http://www.elsevier.com/xml/common/dtd}title')
            if title is not None and title.text:
                text_parts.append(title.text.strip())
            
            # Extract abstract if available
            abstract = article.find('.//{http://www.elsevier.com/xml/common/dtd}abstract')
            if abstract is not None:
                extract_text_from_element(abstract, text_parts)
            
            # Extract sections (main body content)
            sections = article.findall('.//{http://www.elsevier.com/xml/common/dtd}sections')
            for section in sections:
                extract_text_from_element(section, text_parts)
            
            # Extract paragraphs
            paras = article.findall('.//{http://www.elsevier.com/xml/common/dtd}simple-para')
            for para in paras:
                extract_text_from_element(para, text_parts)
            
            # Extract list items
            list_items = article.findall('.//{http://www.elsevier.com/xml/common/dtd}list-item')
            for item in list_items:
                extract_text_from_element(item, text_parts)
        else:
            # Fallback: extract all text from root, but filter out metadata
            for elem in root.iter():
                tag = elem.tag
                # Skip metadata elements
                if any(skip in tag for skip in ['xocs:', 'coredata', 'objects', 'ref-info', 
                                                'attachment', 'link', 'object', 'meta']):
                    continue
                
                # Get text content
                if elem.text and elem.text.strip():
                    # Only add if it looks like actual content (not just metadata)
                    text = elem.text.strip()
                    if len(text) > 10 and not text.startswith('http') and not text.startswith('doi:'):
                        text_parts.append(text)
        
        # Join all text parts
        full_text = ' '.join(text_parts)
        
        # Clean up the text
        # Remove excessive whitespace
        full_text = re.sub(r'\s+', ' ', full_text)
        # Remove URLs
        full_text = re.sub(r'https?://\S+', '', full_text)
        # Remove email addresses
        full_text = re.sub(r'\S+@\S+', '', full_text)
        # Remove XML entities
        full_text = re.sub(r'&[a-zA-Z]+;', '', full_text)
        # Remove special characters that might be artifacts
        full_text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\']+', ' ', full_text)
        # Clean up multiple spaces again
        full_text = re.sub(r'\s+', ' ', full_text)
        full_text = full_text.strip()
        
        return full_text
        
    except ET.ParseError as e:
        print(f"XML parsing error in {xml_file}: {e}")
        # Fallback to regex-based extraction
        return extract_text_regex(xml_file)
    except Exception as e:
        print(f"Error processing {xml_file}: {e}")
        return ""

def extract_text_regex(xml_file: Path) -> str:
    """
    Fallback method: extract text using regex if XML parsing fails.
    """
    try:
        with open(xml_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Remove all XML tags
        text = re.sub(r'<[^>]+>', ' ', content)
        
        # Remove XML declarations and processing instructions
        text = re.sub(r'<\?[^>]+\?>', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove XML entities
        text = re.sub(r'&[a-zA-Z]+;', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove lines that look like metadata (DOIs, IDs, etc.)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                # Skip lines that are mostly metadata
                if not (line.startswith('doi:') or 
                       line.startswith('http') or
                       re.match(r'^[A-Z0-9\-]+$', line) or
                       'xmlns' in line.lower()):
                    cleaned_lines.append(line)
        
        text = ' '.join(cleaned_lines)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
        
    except Exception as e:
        print(f"Error in regex extraction for {xml_file}: {e}")
        return ""

def process_xml_file(xml_file: Path, output_dir: Optional[Path] = None) -> bool:
    """
    Process a single XML file and save extracted text.
    """
    try:
        # Extract text
        text = extract_article_text(xml_file)
        
        if not text or len(text) < 100:  # Skip files with too little content
            print(f"Warning: {xml_file.name} produced very little text ({len(text)} chars)")
            return False
        
        # Determine output path
        if output_dir is None:
            output_dir = xml_file.parent
        
        # Create output filename
        output_file = output_dir / f"{xml_file.stem}.txt"
        
        # Write text to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"✓ Processed: {xml_file.name} -> {output_file.name} ({len(text)} chars)")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {xml_file.name}: {e}")
        return False

def process_directory(input_dir: Path, output_dir: Optional[Path] = None, recursive: bool = True) -> None:
    """
    Process all XML files in a directory.
    """
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all XML files
    if recursive:
        xml_files = list(input_dir.rglob("*.xml"))
    else:
        xml_files = list(input_dir.glob("*.xml"))
    
    if not xml_files:
        print(f"No XML files found in {input_dir}")
        return
    
    print(f"\nFound {len(xml_files)} XML files in {input_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Process each file
    success_count = 0
    for i, xml_file in enumerate(xml_files, 1):
        if process_xml_file(xml_file, output_dir):
            success_count += 1
        
        if i % 10 == 0:
            print(f"Progress: {i}/{len(xml_files)} files processed...")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count}/{len(xml_files)} files")
    print(f"{'='*60}\n")

def main():
    """
    Main function to process XML files from seed directories.
    """
    if len(sys.argv) > 1:
        # Process specific directory provided as argument
        input_dir = Path(sys.argv[1])
        output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
        process_directory(input_dir, output_dir)
    else:
        # Process all seed subdirectories
        seed_dir = Path(__file__).parent / "seed"
        
        if not seed_dir.exists():
            print(f"Error: seed directory not found at {seed_dir}")
            sys.exit(1)
        
        # Process each subdirectory
        subdirs = [d for d in seed_dir.iterdir() if d.is_dir()]
        
        print(f"Processing {len(subdirs)} seed subdirectories...\n")
        
        for subdir in subdirs:
            print(f"\n{'='*60}")
            print(f"Processing: {subdir.name}")
            print(f"{'='*60}")
            process_directory(subdir, output_dir=None, recursive=False)

if __name__ == "__main__":
    main()

