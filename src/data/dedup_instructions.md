Step-by-Step Instructions for Using the Deduplication Repository

Here's a comprehensive guide to preparing your data and using the Google dataset 

deduplication repository for deduplicating your text data:
1. Setup and Installation

Clone the repository:
bashgit clone https://github.com/google-research/deduplicate-text-datasets.git
cd `deduplicate-text-datasets`

Install Rust (if not already installed):
`bashcurl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

Install C compiler (if not already installed):
`bashsudo apt-get install gcc`

Install Python dependencies:
`bashpip3 install numpy scipy sentencepiece`
`pip3 install -r requirements-tf.txt`

Build the Rust code:
`bashcargo build`


2. Preparing Your Data
Choose one of the following methods depending on your data format:
A. For a Single Text File
If your data is already in a single file, you can skip directly to step 3.
B. For Multiple Text Files

Create a directory to store your processed data:
`bashmkdir processed_data`

Combine all your text files into a single file with proper separators:
`bashpython3 scripts/load_dataset_hf.py --save_dir processed_data --name your_dataset_name --data_dir /path/to/your/text/files --split all`
This will create a file processed_data/your_dataset_name.all and a corresponding size file processed_data/your_dataset_name.all.size.

C. For Hugging Face Datasets

Load and prepare the dataset:
`bashpython3 scripts/load_dataset_hf.py --save_dir processed_data --name your_dataset_name --split train`
You can replace train with any other split you want to deduplicate.

D. For TensorFlow Datasets

Load and prepare the dataset:
`bashpython3 scripts/load_dataset.py --data_dir /path/to/tensorflow_datasets --save_dir processed_data --name your_dataset_name --split train`


3. Creating the Suffix Array

Once you have your data in a single file, create the suffix array:
`bashpython3 scripts/make_suffix_array.py processed_data/your_dataset_name.all`
If you're working with a very large dataset, you might need to increase the file limit:
bashulimit -Sn 1000000
This will create a file processed_data/your_dataset_name.all.table.bin.

4. Finding Duplicates
Identify substrings of a given length that are repeated in your dataset:
`bashcargo run self-similar --data-file processed_data/your_dataset_name.all --length-threshold 100 --cache-dir /tmp/cache --num-threads 8`
Important parameters:
!!--length-threshold: Minimum sequence length (in bytes) to consider as a duplicate. If you're using tokenized text, set this to double (e.g., 100 bytes = 50 tokens).
!!--num-threads: Number of CPU cores to use. For large datasets, use as many as available.

This will create cache files in /tmp/cache/.

5. Collecting Duplicates
Merge the identified duplicate sequences into byte ranges:
`bashcargo run collect --data-file processed_data/your_dataset_name.all --cache-dir /tmp/cache --length-threshold 100 > /tmp/remove_ranges.byterange`
This will output byte ranges to /tmp/remove_ranges.byterange.

6. Removing Duplicates
Use the deduplicate_single_file.sh script to create a deduplicated version of your data:
`bashbash scripts/deduplicate_single_file.sh processed_data/your_dataset_name.all processed_data/your_dataset_name.all.deduplicated 100 8`
Parameters:

Input file path
Output file path
Length threshold (same as used in finding duplicates)
Number of CPU cores

7. Optional: Second Deduplication Pass
For thorough deduplication, run the process a second time on the deduplicated data:
`bashbash scripts/deduplicate_single_file.sh processed_data/your_dataset_name.all.deduplicated processed_data/your_dataset_name.all.deduplicated.final 100 8`

8. Verifying Deduplication
Check if the deduplication was successful by counting duplicates in the final file:
`bashpython3 scripts/load_dataset.py --data_dir . --save_dir verification --name your_dataset_name.deduplicated.final --split all`
`python3 scripts/make_suffix_array.py verification/your_dataset_name.deduplicated.final.all`
`cargo run self-similar --data-file verification/your_dataset_name.deduplicated.final.all --length-threshold 100 --cache-dir /tmp/cache2 --num-threads 8`
The number of duplicates found should be significantly lower, ideally close to zero.
Additional Options and Configurations
Tokenizing Data
If your dataset is large, you can tokenize it to reduce size:
bashpython3 scripts/load_dataset.py --data_dir /path/to/data --save_dir processed_data --name your_dataset_name --split train --tokenize
When working with tokenized data, remember to adjust the length-threshold (double the number of tokens).
Finding Duplicates Between Two Datasets
To find duplicates between two datasets:
bashcargo run across-similar --data-file-1 dataset1.txt --data-file-2 dataset2.txt --length-threshold 100 --cache-dir /tmp/cache --num-threads 8
Counting Specific Sequences
To check how many times a specific sequence appears:
bashpython3 scripts/count_occurrences.py --suffix processed_data/your_dataset_name.all --query "your sequence"
Important Notes

The memory requirements scale with dataset size:

Small datasets (<10GB): ~16GB RAM
C4-sized datasets (~300GB): >600GB RAM and >1TB disk space


The deduplication process removes all identified duplicates, which may break sentence flow in some cases. This usually doesn't harm language model training since relatively little text is removed.
If you're deduplicating a dataset in a specific format (e.g., TensorFlow Dataset), you might need to write custom code to post-process the deduplicated output back into your desired format.
For very large datasets, consider running on a machine with many CPU cores to speed up the process.
The length-threshold parameter is critical - too short and you'll remove too much data, too long and you'll miss duplicates.
Always run the deduplication process twice for best results, as new duplicates can emerge after the first pass.