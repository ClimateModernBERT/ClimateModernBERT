import os
import numpy as np
import json
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import openparse
from openparse import processing
import re


# https://github.com/Filimoa/open-parse
############################################################
####################### Parser #############################
############################################################

class DocumentProcessor:
    def __init__(self, input_dir, output_dir, use_semantic_processing=False, openai_api_key=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
    
        if use_semantic_processing and openai_api_key:
            print("USE semantic processing pipeline")
            semantic_pipeline = processing.SemanticIngestionPipeline(
                openai_api_key=openai_api_key,
                model="xxx", # wait
                min_tokens=64,
                max_tokens=1024,
            )
            self.parser = openparse.DocumentParser(processing_pipeline=semantic_pipeline)
        else:
            print("Using default processing pipeline")
            self.parser = openparse.DocumentParser()
    
    def process_documents(self, file_types=None, filter_topics=None):
        """
        Process all documents in the input directory using OpenParse.
        
        Args:
            file_types: List of file extensions to process (e.g., ['.pdf', '.xml'])
            filter_topics: List of topics to filter content (e.g., ['climate', 'emission'])
        """
        if file_types is None:
            file_types = ['.pdf', '.xml', '.docx', '.html', '.txt']
            
        if filter_topics is None:
            filter_topics = []
        
        files = []
        for ext in file_types:
            files.extend(list(self.input_dir.glob(f'**/*{ext}')))
        
        print(f"Found {len(files)} files to process")
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(tqdm(
                pool.imap(
                    self._process_file_wrapper,
                    [(file_path, filter_topics) for file_path in files]
                ),
                total=len(files),
                desc="Processing documents"
            ))
        
        successful = sum(1 for r in results if r)
        print(f"Successfully processed {successful} out of {len(files)} files")
    
    def _process_file_wrapper(self, args):
        file_path, filter_topics = args
        return self.process_file(file_path, filter_topics)
    
    def process_file(self, file_path, filter_topics):
        # Parse the document using OpenParse
        parsed_doc = self.parser.parse(str(file_path))
        
        # Extract text from all nodes
        extracted_content = []
        
        for node in parsed_doc.nodes:
            # Create a structured record for each node
            node_content = {
                "text": node.text if hasattr(node, "text") else "",
                "type": node.__class__.__name__,
                "page": getattr(node, "page", None),
                "is_title": getattr(node, "is_title", False),
                "is_heading": getattr(node, "tag_name", "").startswith("h") if hasattr(node, "tag_name") else False,
                "metadata": {}
            }
            
            # Add additional metadata if available
            if hasattr(node, "tag_name"):
                node_content["metadata"]["tag_name"] = node.tag_name
            
            if hasattr(node, "bbox"):
                node_content["metadata"]["bbox"] = node.bbox
            
            # Apply topic filtering if specified
            if filter_topics:
                if any(topic.lower() in node_content["text"].lower() for topic in filter_topics):
                    extracted_content.append(node_content)
            else:
                extracted_content.append(node_content)
        
        # Save the extracted content
        output_file = self.output_dir / f"{file_path.stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "filename": file_path.name,
                "filepath": str(file_path),
                "nodes": extracted_content
            }, f, ensure_ascii=False, indent=2)
        
        # Also save plain text for the deduplication process
        text_output = self.output_dir / f"{file_path.stem}.txt"
        with open(text_output, 'w', encoding='utf-8') as f:
            # Combine text from all nodes with proper separation
            text_content = "\n\n".join([node["text"] for node in extracted_content if node["text"]])
            f.write(text_content)
        
        return True
