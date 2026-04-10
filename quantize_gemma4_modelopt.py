"""Quantize Gemma4 26B-A4B to NVFP4 using NVIDIA modelopt.

Follows NVIDIA's Gemma-4-31B-IT-NVFP4 recipe:
1. Load the unquantized model
2. Exclude self_attn layers from quantization (keep as BF16)
3. Quantize MoE experts and dense layers to NVFP4
4. Export in modelopt format compatible with vLLM

Usage:
    python3 quantize_gemma4_modelopt.py \
        --input /root/models/gemma-4-26B-A4B-it-original \
        --output /root/models/gemma-4-26B-A4B-it-NVFP4-ours \
        --calib-size 512
"""

import argparse
import torch
import gc
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/root/models/gemma-4-26B-A4B-it-original")
    parser.add_argument("--output", default="/root/models/gemma-4-26B-A4B-it-NVFP4-ours")
    parser.add_argument("--calib-size", type=int, default=512,
                        help="Number of calibration samples")
    parser.add_argument("--calib-dataset", default="cnn_dailymail",
                        help="HuggingFace dataset for calibration")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Calibration: {args.calib_size} samples from {args.calib_dataset}")

    # Import modelopt
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_tensorrt_llm_checkpoint
    print(f"modelopt version: {mtq.__version__ if hasattr(mtq, '__version__') else 'unknown'}")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.input, trust_remote_code=True)

    print(f"Loading model (this will use ~52GB RAM)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.input,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

    # Build calibration dataset
    print(f"\nPreparing calibration data...")
    from datasets import load_dataset
    try:
        dataset = load_dataset(args.calib_dataset, "3.0.0", split="train",
                               streaming=True)
    except Exception:
        dataset = load_dataset(args.calib_dataset, split="train", streaming=True)

    calib_data = []
    for i, sample in enumerate(dataset):
        if i >= args.calib_size:
            break
        text = sample.get("article", sample.get("text", ""))
        if len(text) > 100:
            tokens = tokenizer(text, max_length=args.max_seq_len,
                               truncation=True, return_tensors="pt")
            calib_data.append(tokens.input_ids.to(args.device))

    print(f"Calibration samples: {len(calib_data)}")

    def calib_loop(model):
        """Run calibration forward passes."""
        model.eval()
        with torch.no_grad():
            for i, input_ids in enumerate(calib_data):
                if i % 50 == 0:
                    print(f"  Calibration {i}/{len(calib_data)}...")
                model(input_ids)

    # Build exclude list — same pattern as NVIDIA 31B
    num_layers = model.config.text_config.num_hidden_layers
    exclude_modules = ["lm_head"]
    for i in range(num_layers):
        exclude_modules.append(f"model.language_model.layers.{i}.self_attn*")
    # Also exclude vision/audio towers
    exclude_modules.extend(["model.vision_tower*", "model.audio_tower*",
                            "model.embed_vision*", "model.embed_audio*"])

    print(f"\nExcluding {len(exclude_modules)} modules from NVFP4")
    print(f"  (all self_attn layers, vision/audio towers, lm_head)")

    # Configure NVFP4 quantization — MLP only (NVIDIA's approach)
    # This quantizes MoE experts + dense FFN, keeps attention in BF16
    quant_config = mtq.NVFP4_MLP_ONLY_CFG

    print(f"\nQuantizing MLP/MoE layers to NVFP4 (attention stays BF16)...")
    print(f"Config: NVFP4_MLP_ONLY_CFG (algorithm: max)")
    mtq.quantize(model, quant_config, forward_loop=calib_loop)
    print("Quantization complete!")

    # Export
    print(f"\nExporting to {args.output}...")
    os.makedirs(args.output, exist_ok=True)

    # Save using modelopt's export
    try:
        from modelopt.torch.export import export_hf
        export_hf(model, args.output)
        print("Exported via modelopt export_hf")
    except (ImportError, AttributeError):
        # Fallback: save with transformers
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        print("Exported via transformers save_pretrained")

    # Write hf_quant_config.json matching NVIDIA's format
    import json
    hf_qconfig = {
        "producer": {"name": "modelopt", "version": "0.42.0"},
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": "FP8",
            "group_size": 16,
            "exclude_modules": exclude_modules,
        }
    }
    with open(os.path.join(args.output, "hf_quant_config.json"), "w") as f:
        json.dump(hf_qconfig, f, indent=2)
    print("Wrote hf_quant_config.json")

    print(f"\nDone! Output: {args.output}")
    print(f"Size: {sum(os.path.getsize(os.path.join(args.output, f)) for f in os.listdir(args.output) if f.endswith('.safetensors')) / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
