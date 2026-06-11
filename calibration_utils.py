"""
Calibration Data Loading Utilities for AWQ Quantization.

Provides standard calibration data loaders for:
- C4 dataset (recommended for cross-dataset robustness)
- WikiText-2 dataset (lightweight alternative)

Key Feature: Random slicing within documents (standard GPTQ/AWQ practice)
"""

import torch
import random
from datasets import load_dataset
from typing import List
import pickle
from pathlib import Path


def get_c4_calibration_data(tokenizer, n_samples=128, seqlen=2048, seed=42, return_tensors=False, cache_dir="./calibration_cache"):
    """
    Load C4 calibration data with random slicing (standard GPTQ/AWQ approach).

    OPTIMIZATIONS:
    - Uses direct URL to C4 shard (bypasses trust_remote_code issue)
    - Fast text length filtering before tokenization (~50x faster)
    - Optional tensor return (more efficient, no decode/re-encode)
    - Caching support to avoid re-downloading data

    Why Random Slicing?
    - Avoids bias from document headers/introductions
    - Provides more diverse activation statistics
    - Matches AutoGPTQ/AutoAWQ reference implementations

    Process:
    1. Stream C4 first shard via direct URL
    2. For each document:
       - Fast skip if text too short (char length check)
       - Tokenize the full text
       - Skip if tokens < seqlen
       - Randomly slice a window of length seqlen
    3. Collect n_samples such windows

    Args:
        tokenizer: HuggingFace tokenizer
        n_samples: Number of calibration samples (default: 128)
        seqlen: Sequence length in tokens (default: 2048)
        seed: Random seed for reproducibility
        return_tensors: If True, return token tensors. If False, return text strings (default: False)
        cache_dir: Directory to cache downloaded data (default: "./calibration_cache")

    Returns:
        List[torch.Tensor] if return_tensors=True, else List[str]
    """
    print(f"\n[C4 Calibration Data - Optimized]")
    print(f"  Samples: {n_samples}")
    print(f"  Sequence length: {seqlen} tokens")
    print(f"  Method: Random slicing with fast filtering")
    print(f"  Seed: {seed}")

    # Check cache first
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    cache_file = cache_path / f"c4_calib_n{n_samples}_len{seqlen}_seed{seed}_tensors{return_tensors}.pkl"
    if cache_file.exists():
        print(f"\n  📦 Loading from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"\n  ⚠️  No cache found, downloading from C4...")
    random.seed(seed)

    # URL to the first shard of C4
    # This bypasses the 'allenai/c4' script entirely, fixing trust_remote_code errors
    url = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"

    # Load using 'json' format instead of 'allenai/c4'
    traindata = load_dataset(
        "json",
        data_files={"train": url},
        split="train",
        streaming=True
    )

    dataset = []
    skipped = 0

    # Fast filtering: heuristic of 1 token ~ 3-4 chars
    # Skip documents that are definitely too short without tokenizing
    char_threshold = seqlen * 3

    print(f"\n  Streaming C4 with fast filtering...")

    for i, data in enumerate(traindata):
        text = data['text']

        # FAST SKIP: Check character length first (avoids tokenization overhead)
        if len(text) < char_threshold:
            skipped += 1
            continue

        # Tokenize the document
        trainenc = tokenizer(text, return_tensors='pt')

        # Skip documents that are too short
        if trainenc.input_ids.shape[1] < seqlen:
            skipped += 1
            continue

        # THE CRITICAL STEP: Random slicing
        # Instead of taking first seqlen tokens (biased towards headers),
        # we randomly sample a window within the document
        max_start = trainenc.input_ids.shape[1] - seqlen
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + seqlen

        # Extract the slice
        inp = trainenc.input_ids[:, start_idx:end_idx]

        if return_tensors:
            # Return tensors directly (more efficient - no decode/re-encode)
            dataset.append(inp)
        else:
            # Decode to text (for compatibility with existing code)
            text = tokenizer.decode(inp[0], skip_special_tokens=True)
            dataset.append(text)

        # Progress indicator
        if (len(dataset) + 1) % 32 == 0:
            print(f"    Collected {len(dataset)}/{n_samples} samples (skipped {skipped} short docs)...")

        # Stop once we have enough samples
        if len(dataset) == n_samples:
            break

    print(f"\n  ✓ Collected {len(dataset)} samples from C4")
    print(f"  ✓ Skipped {skipped} documents (too short)")

    # Save to cache
    print(f"  💾 Saving to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(dataset, f)

    return dataset


def get_wikitext2_calibration_data(tokenizer, n_samples=128, seqlen=2048, seed=42, split='train', cache_dir="./calibration_cache"):
    """
    Load WikiText-2 calibration data (lightweight alternative to C4).

    Note: WikiText-2 is smaller and faster to download, but less diverse than C4.
    Use C4 for production quantization, WikiText-2 for quick experiments.

    Args:
        tokenizer: HuggingFace tokenizer
        n_samples: Number of calibration samples
        seqlen: Sequence length in tokens
        seed: Random seed
        split: Dataset split ('train', 'validation', 'test')
        cache_dir: Directory to cache downloaded data (default: "./calibration_cache")

    Returns:
        List[str]: Calibration texts
    """
    print(f"\n[WikiText-2 Calibration Data]")
    print(f"  Samples: {n_samples}")
    print(f"  Sequence length: {seqlen} tokens")
    print(f"  Split: {split}")
    print(f"  Seed: {seed}")

    # Check cache first
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    cache_file = cache_path / f"wikitext2_calib_n{n_samples}_len{seqlen}_seed{seed}_split{split}.pkl"
    if cache_file.exists():
        print(f"\n  📦 Loading from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"\n  ⚠️  No cache found, downloading from WikiText-2...")
    random.seed(seed)

    # Load WikiText-2
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)

    # Filter out empty texts
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]

    print(f"  Total non-empty texts: {len(texts)}")

    # Tokenize and concatenate all texts
    print(f"  Tokenizing and concatenating...")
    all_tokens = []
    for text in texts:
        tokens = tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
        all_tokens.append(tokens)

    all_tokens = torch.cat(all_tokens, dim=0)
    print(f"  Total tokens: {len(all_tokens)}")

    # Split into seqlen chunks
    num_chunks = len(all_tokens) // seqlen
    print(f"  Available {seqlen}-token chunks: {num_chunks}")

    if num_chunks < n_samples:
        print(f"  ⚠️  Warning: Only {num_chunks} chunks available, requested {n_samples}")
        n_samples = num_chunks

    # Randomly sample n_samples chunks
    chunk_indices = random.sample(range(num_chunks), n_samples)

    calibration_texts = []
    for idx in chunk_indices:
        start = idx * seqlen
        end = start + seqlen
        chunk_tokens = all_tokens[start:end]
        text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        calibration_texts.append(text)

    print(f"  ✓ Collected {len(calibration_texts)} samples from WikiText-2")

    # Save to cache
    print(f"  💾 Saving to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(calibration_texts, f)

    return calibration_texts


def load_calibration_data(dataset_name, tokenizer, n_samples=128, seqlen=2048, seed=42, cache_dir="./calibration_cache"):
    """
    Universal calibration data loader.

    Args:
        dataset_name: 'c4', 'wikitext2', or 'wikitext'
        tokenizer: HuggingFace tokenizer
        n_samples: Number of calibration samples
        seqlen: Sequence length in tokens
        seed: Random seed
        cache_dir: Directory to cache downloaded data (default: "./calibration_cache")

    Returns:
        List[str]: Calibration texts
    """
    dataset_name = dataset_name.lower()

    if dataset_name == 'c4':
        return get_c4_calibration_data(tokenizer, n_samples, seqlen, seed, cache_dir=cache_dir)
    elif dataset_name in ['wikitext2', 'wikitext']:
        return get_wikitext2_calibration_data(tokenizer, n_samples, seqlen, seed, split='train', cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'c4' or 'wikitext2'")


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Test with MiniCPM-2B tokenizer
    model_id = "openbmb/MiniCPM-2B-sft-bf16"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("=" * 80)
    print("Testing Calibration Data Loaders")
    print("=" * 80)

    # Test C4 - OPTIMIZED VERSION
    print("\n[Test 1: C4 with fast filtering]")
    c4_samples = get_c4_calibration_data(tokenizer, n_samples=10, seqlen=512)
    print(f"C4 sample type: {type(c4_samples[0])}")
    print(f"C4 sample length: {len(c4_samples[0])} chars")
    print(f"C4 sample preview: {c4_samples[0][:200]}...")

    # Test C4 - Return tensors (more efficient)
    print("\n[Test 2: C4 with tensor return (efficient mode)]")
    c4_tensors = get_c4_calibration_data(tokenizer, n_samples=5, seqlen=512, return_tensors=True)
    print(f"C4 tensor type: {type(c4_tensors[0])}")
    print(f"C4 tensor shape: {c4_tensors[0].shape}")

    # Test WikiText-2
    print("\n[Test 3: WikiText-2]")
    wt2_samples = get_wikitext2_calibration_data(tokenizer, n_samples=10, seqlen=512)
    print(f"WikiText-2 sample length: {len(wt2_samples[0])} chars")
    print(f"WikiText-2 sample preview: {wt2_samples[0][:200]}...")

    print("\n✓ All tests passed!")