import argparse
import json
import pandas as pd
from datasets import Dataset
import torch
from tqdm import tqdm
import random
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from chat import LocalChat
from vllm_utils import MODEL_DICT, GENE_ARGS_DICT
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")


STYLES = {
    "public_awareness": """Based on this climate news excerpt:
"<EXTRACT>"

Create an accessible article about climate change that:
- Explains Complex Concepts: Break down scientific terms into everyday language
- Local Relevance: Connect global climate issues to local community impacts
- Practical Actions: Suggest concrete steps individuals can take
- Human Stories: Include relatable examples and potential human impacts

Write in an engaging, informative tone suitable for general public awareness. Focus on making climate science understandable and actionable.""",

    "industry_perspective": """Using this climate-related news as context:
"<EXTRACT>"

Develop a comprehensive industry analysis covering:
- Sector Impact: How different industries are affected by or contributing to climate change
- Innovation & Technology: Emerging solutions and technological adaptations
- Business Strategy: Corporate responses and sustainable business models
- Investment Trends: Green finance and ESG considerations

Write from a business and industry perspective, highlighting opportunities and challenges.""",

    "environmental_journalism": """Drawing from this climate news piece:
"<EXTRACT>"

Create an in-depth environmental journalism article that:
- Investigative Depth: Explore underlying causes and systemic issues
- Ecosystem Impact: Detail effects on biodiversity, habitats, and natural systems
- Climate Justice: Address equity and vulnerable population concerns
- Solution Spotlight: Highlight successful interventions and best practices

Write in an investigative journalism style that combines factual reporting with compelling narrative.""",

}

EXTRACT_SIZE = 800

def get_args():
    parser = argparse.ArgumentParser(description="Generate synthetic climate news data")
    parser.add_argument("--input_file", type=str, default="../seed/climatenews_2000_filtered.csv",
                       help="Input CSV file with climate news (path relative to script or absolute)")
    parser.add_argument("--output_file", type=str, default="synthetic_climate_news.jsonl",
                       help="Output JSONL file for synthetic data")
    parser.add_argument("--generation_style", type=str, default="public_awareness",
                       choices=list(STYLES.keys()),
                       help="Style of synthetic generation")
    parser.add_argument("--run_all_styles", action="store_true",
                       help="Generate using all available styles")
    parser.add_argument("--samples_per_seed", type=int, default=5,
                       help="Number of synthetic samples to generate per seed")
    parser.add_argument("--max_seeds", type=int, default=None,
                       help="Maximum number of seed articles to use (None = use all)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                       help="Hugging Face model to use (Qwen3-30B-A3B-Instruct-2507)")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="HuggingFace token for accessing private/gated models (or set HF_TOKEN env var)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run model on")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p for generation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--cache_dir", type=str, required=True,
                       help="Cache directory for storing model outputs")
    parser.add_argument("--n_threads", type=int, default=4,
                       help="Number of threads for concurrent generation")
    parser.add_argument("--hf_repo", type=str, default=None,
                       help="HuggingFace repository name to push dataset (e.g., 'username/dataset-name')")
    return parser.parse_args()

def load_climate_news(file_path):
    try:
        if not os.path.isabs(file_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, file_path)
        
        df = pd.read_csv(file_path)
        df = df[df['text_clean'].notna() & (df['text_clean'].str.len() > 100)]
        print(f"Loaded {len(df)} valid climate news articles from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def build_prompt(text, style="analytical_report"):
    snippet = text.strip()
    snippet = snippet[:min(len(snippet), EXTRACT_SIZE)]
    prompt = STYLES[style].replace("<EXTRACT>", snippet)
    return prompt

def generate_single_sample(chat_client, prompt, sample_idx):
    """Generate a single sample from the chat client."""
    response, _ = chat_client.ask(prompt, sample_idx=sample_idx)
    return response

def main():
    args = get_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"Loading climate news from {args.input_file}...")
    df = load_climate_news(args.input_file)
    if df is None:
        return
    
    if args.max_seeds is None:
        seed_articles = df
        num_seeds = len(df)
    else:
        num_seeds = min(args.max_seeds, len(df))
        seed_articles = df.sample(n=num_seeds, random_state=42)
    
    unique_seeds = seed_articles['text_clean'].nunique()
    print(f"📊 Seed Statistics:")
    print(f"   - Total seeds selected: {num_seeds}")
    print(f"   - Unique seed texts: {unique_seeds}")
    if unique_seeds < num_seeds:
        print(f"   ⚠️  Warning: {num_seeds - unique_seeds} duplicate seed texts detected")
    print(f"   - Each seed will generate {args.samples_per_seed} synthetic samples")
    
    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    print(f"📦 Using cache directory: {cache_dir}")
    
    print(f"Initializing LocalChat with model: {args.model_name}...")
    
    # Determine the model name for the client and the key for generation args
    model_name_for_client = MODEL_DICT.get(args.model_name, args.model_name)
    model_key = args.model_name
    # If full name was provided, try to find the short key for GENE_ARGS_DICT
    if args.model_name in MODEL_DICT.values():
        for k, v in MODEL_DICT.items():
            if v == args.model_name:
                model_key = k
                break
    
    # Get base generation args from GENE_ARGS_DICT and update with argparse values
    generation_config = GENE_ARGS_DICT.get(model_key, {}).copy()
    generation_config.update({
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    })
    
    system_prompt = "You are an expert climate journalist and researcher capable of generating high-quality, factual content about climate change based on real news excerpts."
    
    try:
        chat_client = LocalChat(
            model=model_name_for_client,
            cache_path=cache_dir,
            system_prompt=system_prompt,
            generation_config=generation_config,
            temp_perturbation=0.1,
        )
        print("✅ LocalChat initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing LocalChat: {e}")
        return
    
    if args.run_all_styles:
        styles_to_use = list(STYLES.keys())
    else:
        styles_to_use = [args.generation_style]
    
    print(f"Generating synthetic data using {len(styles_to_use)} style(s) with {args.n_threads} threads...")
    
    # Build all tasks upfront
    tasks = []
    used_seed_texts = set()
    
    for seed_idx, (idx, row) in enumerate(tqdm(seed_articles.iterrows(), desc="Building tasks", total=len(seed_articles))):
        seed_text = row['text_clean']
        seed_date = row.get('docdate', '')
        seed_id = idx
        
        seed_text_hash = hash(seed_text[:100])
        if seed_text_hash in used_seed_texts:
            continue
        used_seed_texts.add(seed_text_hash)
        
        for style in styles_to_use:
            prompt = build_prompt(seed_text, style)
            for sample_idx in range(args.samples_per_seed):
                tasks.append({
                    "seed_idx": seed_idx,
                    "seed_id": seed_id,
                    "seed_date": seed_date,
                    "style": style,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                })
    
    print(f"📋 Total tasks to process: {len(tasks)}")
    
    # Process tasks concurrently
    results = {}
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=args.n_threads) as executor:
        futures = {
            executor.submit(generate_single_sample, chat_client, task["prompt"], task["sample_idx"]): task
            for task in tasks
        }
        pbar = tqdm(as_completed(futures), desc="Generating samples", total=len(futures))
        for future in pbar:
            task = futures[future]
            try:
                response = future.result()
                if response:
                    # Use a tuple key to maintain ordering later
                    key = (task["seed_idx"], task["style"], task["sample_idx"])
                    results[key] = {
                        "text": response,
                        "seed_date": task["seed_date"],
                        "seed_id": task["seed_id"],
                        "style": task["style"],
                        "sample_idx": task["sample_idx"],
                    }
                    success_count += 1
                else:
                    error_count += 1
            except Exception as e:
                error_count += 1
                tqdm.write(f"Error generating sample for seed {task['seed_id']}, style {task['style']}: {e}")
            pbar.set_postfix({"✓": success_count, "✗": error_count})
    
    # Sort results by (seed_idx, style, sample_idx) to maintain consistent ordering
    sorted_keys = sorted(results.keys())
    
    all_synthetic_data = []
    synthetic_id = 1
    for key in sorted_keys:
        result = results[key]
        synthetic_entry = {
            "id": synthetic_id,
            "text": result["text"]
        }
        seed_date = result["seed_date"]
        if seed_date and str(seed_date) != 'nan' and str(seed_date) != '':
            synthetic_entry["docdate"] = str(seed_date)
        
        all_synthetic_data.append(synthetic_entry)
        synthetic_id += 1
    
    print(f"\nSaving {len(all_synthetic_data)} synthetic samples to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in all_synthetic_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    
    print(f"✅ Successfully generated {len(all_synthetic_data)} synthetic climate news articles!")
    print(f"📊 Statistics:")
    print(f"   - Seeds used: {num_seeds}")
    print(f"   - Styles used: {len(styles_to_use)}")
    print(f"   - Samples per seed per style: {args.samples_per_seed}")
    print(f"   - Total samples: {len(all_synthetic_data)}")
    
    summary_data = []
    for item in all_synthetic_data:
        summary_data.append({
            "id": item["id"],
            "text_length": len(item["text"]),
            "text_preview": item["text"][:200] + "...",
            "docdate": item.get("docdate", "")
        })
    summary_df = pd.DataFrame(summary_data)
    summary_file = args.output_file.replace('.jsonl', '_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"📋 Summary saved to {summary_file}")

    # Push to HuggingFace if repo is specified
    if args.hf_repo:
        print(f"📤 Pushing dataset to HuggingFace: {args.hf_repo}...")
        try:
            hf_dataset = Dataset.from_list(all_synthetic_data)
            hf_dataset.push_to_hub(
                args.hf_repo,
                private=False,
            )
            print(f"✅ Dataset pushed to HuggingFace: https://huggingface.co/datasets/{args.hf_repo}")
        except Exception as e:
            print(f"❌ Error pushing to HuggingFace: {e}")

if __name__ == "__main__":
    main()
