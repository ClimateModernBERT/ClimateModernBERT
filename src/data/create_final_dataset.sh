#!/bin/bash

echo "=== Creating Final Training Dataset with Proper Deduplication ==="

# Navigate to the processed_data directory
cd processed_data

# Step 1: Combine all deduplicated files
echo "Step 1: Combining all *_deduplicated.txt files from processed_data..."
cat *_deduplicated.txt > combined_temp.txt

echo "Combined file size: $(du -h combined_temp.txt | cut -f1)"
echo "Total lines in combined file: $(wc -l < combined_temp.txt)"

# Step 2: Go back to main directory and run deduplication
echo ""
echo "Step 2: Running advanced deduplication using deduplicate-text-datasets..."
cd ..

# Run the deduplication tool on the combined file
python -m deduplicate_text_datasets \
    --path processed_data/combined_temp.txt \
    --save_path training_dataset_xml \
    --cache_dir ./cache_final \
    --use_hashing true \
    --num_perm 64 \
    --threshold 0.7

# Step 3: Clean up temporary file
echo ""
echo "Step 3: Cleaning up temporary files..."
rm processed_data/combined_temp.txt

# Step 4: Show final results
echo ""
echo "=== Final Results ==="
echo "Final deduplicated dataset: training_dataset_xml"
if [ -f "training_dataset_xml" ]; then
    echo "Final file size: $(du -h training_dataset_xml | cut -f1)"
    echo "Final line count: $(wc -l < training_dataset_xml)"
else
    echo "Error: training_dataset_xml not created!"
fi

echo "Deduplication complete!"
