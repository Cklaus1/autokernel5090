#!/usr/bin/env python3
"""Convert Gemma 4 26B A4B FP8 → NVFP4 using NVIDIA ModelOpt.

Workflow:
1. Load FP8 model with transformers (auto-dequantizes to BF16)
2. Apply NVFP4 quantization via modelopt with calibration
3. Export HF checkpoint compatible with vLLM
"""

import torch
import gc
import os
from pathlib import Path

# Config
SRC_MODEL = "/root/models/gemma-4-26B-A4B-it-FP8"
OUTPUT_DIR = "/root/models/gemma-4-26B-A4B-it-NVFP4"
CALIB_SAMPLES = 512
CALIB_SEQ_LEN = 512

def get_calib_dataloader(tokenizer, num_samples=CALIB_SAMPLES, seq_len=CALIB_SEQ_LEN):
    """Create calibration dataloader from random data or a standard dataset."""
    from datasets import load_dataset

    # Use C4 for calibration (standard choice)
    try:
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        samples = []
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            samples.append(example["text"])
    except Exception:
        print("WARNING: Could not load C4 dataset, using random calibration data")
        samples = ["The quick brown fox jumps over the lazy dog. " * 50] * num_samples

    # Tokenize
    batch = tokenizer(
        samples,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seq_len,
    )

    # Create simple dataloader
    class CalibDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __len__(self):
            return self.encodings["input_ids"].shape[0]
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.encodings.items()}

    dataset = CalibDataset(batch)
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)


def forward_loop(model):
    """Calibration forward loop for modelopt."""
    print(f"Running calibration with {CALIB_SAMPLES} samples, seq_len={CALIB_SEQ_LEN}...")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SRC_MODEL)
    dataloader = get_calib_dataloader(tokenizer)

    device = next(model.parameters()).device

    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)

        if (i + 1) % 100 == 0:
            print(f"  Calibrated {i+1}/{CALIB_SAMPLES} samples")

    print("Calibration complete.")


def main():
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Source: {SRC_MODEL}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Step 1: Load model — FP8 compressed-tensors will dequantize to BF16
    print("\n[1/4] Loading FP8 model (dequantizes to BF16)...")
    model = AutoModelForCausalLM.from_pretrained(
        SRC_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Will split across GPU + CPU if needed
        trust_remote_code=True,
    )
    model.eval()

    # Check memory
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU memory used: {allocated:.1f} GB")

    # Step 2: Quantize with NVFP4
    print("\n[2/4] Applying NVFP4 quantization with modelopt...")
    quant_cfg = mtq.NVFP4_DEFAULT_CFG
    print(f"  Config: algorithm={quant_cfg['algorithm']}")

    model = mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: Print summary
    print("\n[3/4] Quantization summary:")
    mtq.print_quant_summary(model)

    # Step 4: Export
    print(f"\n[4/4] Exporting to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    export_hf_checkpoint(
        model,
        dtype=torch.bfloat16,
        export_dir=OUTPUT_DIR,
    )

    # Copy tokenizer and config files
    tokenizer = AutoTokenizer.from_pretrained(SRC_MODEL)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Copy additional config files
    import shutil
    for fname in ["chat_template.jinja", "processor_config.json", "generation_config.json"]:
        src = Path(SRC_MODEL) / fname
        if src.exists():
            shutil.copy2(src, Path(OUTPUT_DIR) / fname)

    print(f"\nDone! Model saved to {OUTPUT_DIR}")
    print(f"Output size: {sum(f.stat().st_size for f in Path(OUTPUT_DIR).rglob('*') if f.is_file()) / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
