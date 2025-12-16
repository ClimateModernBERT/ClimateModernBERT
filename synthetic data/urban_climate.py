import argparse
import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import random
from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

STYLES = {
    "public_awareness": """Based on this climate research excerpt:
"<EXTRACT>"

Create an accessible article about climate change that:
- Explains Complex Concepts: Break down scientific terms into everyday language
- Local Relevance: Connect global climate issues to local community impacts
- Practical Actions: Suggest concrete steps individuals can take
- Human Stories: Include relatable examples and potential human impacts

Write in an engaging, informative tone suitable for general public awareness. Focus on making climate science understandable and actionable.""",
    "industry_perspective": """Using this climate-related research as context:
"<EXTRACT>"

Develop a comprehensive industry analysis covering:
- Sector Impact: How different industries are affected by or contributing to climate change
- Innovation & Technology: Emerging solutions and technological adaptations
- Business Strategy: Corporate responses and sustainable business models
- Investment Trends: Green finance and ESG considerations

Write from a business and industry perspective, highlighting opportunities and challenges.""",
    "environmental_journalism": """Drawing from this climate research piece:
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
    parser = argparse.ArgumentParser(description="Generate synthetic urban climate data")
    parser.add_argument("--input_dir", type=str, default="../seed/UrbanClimate")
    parser.add_argument("--output_file", type=str, default="synthetic_urban_climate.jsonl")
    parser.add_argument("--generation_style", type=str, default="public_awareness", choices=list(STYLES.keys()))
    parser.add_argument("--run_all_styles", action="store_true")
    parser.add_argument("--samples_per_seed", type=int, default=5)
    parser.add_argument("--max_seeds", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    return parser.parse_args()


def load_text_files(input_dir):
    if not os.path.isabs(input_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(script_dir, input_dir)
    input_path = Path(input_dir)
    text_files = list(input_path.glob("*.txt"))
    if not text_files:
        print(f"No .txt files found in {input_dir}")
        return None
    seed_data = []
    for txt_file in text_files:
        try:
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            if len(text) > 100:
                seed_data.append({"filename": txt_file.name, "filepath": str(txt_file), "text": text})
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
            continue
    if not seed_data:
        print(f"No valid text files found in {input_dir}")
        return None
    df = pd.DataFrame(seed_data)
    print(f"Loaded {len(df)} valid text files from {input_dir}")
    return df


def extract_text_context(text):
    text = text.strip()
    if len(text) <= EXTRACT_SIZE:
        return text
    return text[:EXTRACT_SIZE]


def build_prompt(text, style="public_awareness"):
    snippet = extract_text_context(text)
    prompt = STYLES[style].replace("<EXTRACT>", snippet)
    return prompt


def generate_synthetic_data(model, tokenizer, seed_text, style, num_samples, args):
    synthetic_samples = []
    prompt = build_prompt(seed_text, style)
    for _ in range(num_samples):
        messages = [
            {
                "role": "system",
                "content": "You are an expert climate researcher and journalist capable of generating high-quality, factual content about climate change based on real research excerpts.",
            },
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(args.device)
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature + random.uniform(-0.1, 0.1),
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        synthetic_samples.append(response)
    return synthetic_samples


def main():
    args = get_args()
    print(f"Loading text files from {args.input_dir}...")
    df = load_text_files(args.input_dir)
    if df is None:
        return
    if args.max_seeds is None:
        seed_articles = df
        num_seeds = len(df)
    else:
        num_seeds = min(args.max_seeds, len(df))
        seed_articles = df.sample(n=num_seeds, random_state=42)
    unique_seeds = seed_articles["text"].nunique()
    print("📊 Seed Statistics:")
    print(f"   - Total seeds selected: {num_seeds}")
    print(f"   - Unique seed texts: {unique_seeds}")
    if unique_seeds < num_seeds:
        print(f"   ⚠️  Warning: {num_seeds - unique_seeds} duplicate seed texts detected")
    print(f"   - Each seed will generate {args.samples_per_seed} synthetic samples")
    cache_dir = "/cluster/scratch/yongyu/cache"
    os.makedirs(cache_dir, exist_ok=True)
    print(f"📦 Using cache directory: {cache_dir}")
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    token_kwargs = {"token": hf_token} if hf_token else {}
    if hf_token:
        print("🔑 Using HuggingFace token for authentication")
    else:
        print("⚠️  No HuggingFace token provided - model must be public or you must be logged in")
    print(f"Loading model: {args.model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=cache_dir, **token_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
            device_map="auto" if args.device == "cuda" else None,
            **token_kwargs,
        )
        if args.device == "cpu":
            model = model.to(args.device)
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    if args.run_all_styles:
        styles_to_use = list(STYLES.keys())
    else:
        styles_to_use = [args.generation_style]
    all_synthetic_data = []
    synthetic_id = 1
    print(f"Generating synthetic data using {len(styles_to_use)} style(s)...")
    used_seed_texts = set()
    for seed_idx, (idx, row) in enumerate(
        tqdm(seed_articles.iterrows(), total=len(seed_articles), desc="Processing seeds")
    ):
        seed_text = row["text"]
        seed_filename = row["filename"]
        seed_id = idx
        seed_text_hash = hash(seed_text[:100])
        if seed_text_hash in used_seed_texts:
            print(f"⚠️  Skipping duplicate seed {seed_id} (same text as previous seed)")
            continue
        used_seed_texts.add(seed_text_hash)
        seed_preview = seed_text[:100].replace("\n", " ") + "..."
        print(f"\n{'='*80}")
        print(f"Seed #{seed_idx + 1}/{num_seeds} (ID: {seed_id}, File: {seed_filename})")
        print(f"Seed preview: {seed_preview}")
        print(f"Seed text length: {len(seed_text)} chars")
        print(f"Generating {args.samples_per_seed} synthetic samples from this unique seed...")
        for style in styles_to_use:
            print(f"  → Style: {style}")
            try:
                synthetic_samples = generate_synthetic_data(
                    model, tokenizer, seed_text, style, args.samples_per_seed, args
                )
                for sample_idx, sample in enumerate(synthetic_samples):
                    synthetic_entry = {
                        "id": synthetic_id,
                        "text": sample,
                        "source_file": seed_filename,
                    }
                    all_synthetic_data.append(synthetic_entry)
                    synthetic_id += 1
                    print(
                        f"    ✓ Generated sample {sample_idx + 1}/{args.samples_per_seed} (ID: {synthetic_id - 1})"
                    )
            except Exception as e:
                print(f"Error generating from seed {seed_id}: {e}")
                continue
    print(f"\nSaving {len(all_synthetic_data)} synthetic samples to {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in all_synthetic_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("✅ Successfully generated synthetic urban climate articles!")
    print("📊 Statistics:")
    print(f"   - Seeds used: {num_seeds}")
    print(f"   - Styles used: {len(styles_to_use)}")
    print(f"   - Samples per seed per style: {args.samples_per_seed}")
    print(f"   - Total samples: {len(all_synthetic_data)}")
    summary_data = []
    for item in all_synthetic_data:
        summary_data.append(
            {
                "id": item["id"],
                "text_length": len(item["text"]),
                "text_preview": item["text"][:200] + "...",
                "source_file": item.get("source_file", ""),
            }
        )
    summary_df = pd.DataFrame(summary_data)
    summary_file = args.output_file.replace(".jsonl", "_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"📋 Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
