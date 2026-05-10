#!/usr/bin/env python3
"""Simple token counter for text files"""
import sys, os

if len(sys.argv) != 2:
    print("Usage: python count_tokens.py <filename>")
    sys.exit(1)

file_path = sys.argv[1]
if not os.path.exists(file_path):
    print(f"❌ File '{file_path}' not found!")
    sys.exit(1)

# Get file size
size_mb = os.path.getsize(file_path) / (1024*1024)
print(f"📁 File: {file_path} ({size_mb:.1f} MB)")

# Read and count tokens (simple whitespace split)
try:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    tokens = len([t for t in text.split() if t.strip()])
    print(f"🔢 Tokens: {tokens:,}")
except Exception as e:
    print(f"❌ Error: {e}") 