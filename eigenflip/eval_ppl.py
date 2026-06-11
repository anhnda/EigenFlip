"""eval_ppl.py -- WikiText-2 / C4 perplexity for a saved quantized model.
Accepts multiple datasets, writes ppl.json into the model dir."""
from __future__ import annotations
import argparse, json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


@torch.no_grad()
def eval_ppl(model, tokenizer, dataset, seqlen, device):
    if dataset == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    elif dataset == "c4":
        ds = load_dataset(
            "json",
            data_files={"validation":
                        "https://huggingface.co/datasets/allenai/c4/resolve/main/"
                        "en/c4-validation.00000-of-00008.json.gz"},
            split="validation", streaming=True)
        text = "\n\n".join(d["text"] for _, d in zip(range(1000), ds))
    else:
        raise ValueError(dataset)
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids.to(device)
    n = ids.shape[1] // seqlen
    nlls = []
    for i in range(n):
        chunk = ids[:, i*seqlen:(i+1)*seqlen]
        out = model(chunk, labels=chunk)
        nlls.append(out.loss.float() * seqlen)
    return torch.exp(torch.stack(nlls).sum() / (n*seqlen)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--datasets", nargs="+", default=["wikitext2"],
                   choices=["wikitext2", "c4"])
    p.add_argument("--seqlen", type=int, default=2048)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True).eval()
    results = {}
    for d in args.datasets:
        ppl = eval_ppl(model, tok, d, args.seqlen, device)
        print(f"{d} PPL: {ppl:.4f}")
        results[d] = ppl
    out = os.path.join(args.model_path, "ppl.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
