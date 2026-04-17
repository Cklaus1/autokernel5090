# Mixed NVFP4 + FP8 Recipe for Gemma-4 26B

Implements T2-E from `rtx_pro6000_experiments.md` (lines 188-199): NVFP4 on attention and
MLP expand projections, FP8 dynamic on `down_proj`. Avoids the QKV per-head scale underflow
observed with all-NVFP4 Gemma quantization while keeping memory below a full BF16 baseline.

## Setup Constraint

`llm-compressor` requires `transformers>=4.56.1`. The `setup.py` pin has already been relaxed
from the original `==4.50.x` requirement — no further changes needed before running this recipe.

## Python Recipe

```python
# quantize_gemma4_26b_mixed.py
import copy
from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC, NVFP4
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "google/gemma-4-26b"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)
ds = ds.map(lambda ex: {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)})
ds = ds.map(
    lambda s: tokenizer(s["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH,
                        truncation=True, add_special_tokens=False),
    remove_columns=ds.column_names,
)

# down_proj: FP8 dynamic per-token (activations) + FP8 weight.
# All attention projections + gate_proj/up_proj: NVFP4 weight (group=16) + NVFP4 global act scale.
scheme_fp8 = copy.deepcopy(FP8_DYNAMIC)
scheme_fp8["targets"] = ["re:.*down_proj.*"]

scheme_nvfp4 = copy.deepcopy(NVFP4)
scheme_nvfp4["targets"] = [
    "re:.*self_attn.q_proj.*",
    "re:.*self_attn.k_proj.*",
    "re:.*self_attn.v_proj.*",
    "re:.*self_attn.o_proj.*",
    "re:.*gate_proj.*",
    "re:.*up_proj.*",
]

recipe = QuantizationModifier(
    config_groups={"group_fp8": scheme_fp8, "group_nvfp4": scheme_nvfp4},
    ignore=["lm_head"],
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    output_dir="workspace/gemma4-26b-nvfp4-fp8-mixed",
)
```

## Expected Quality Impact

- NVFP4 on `gate_proj`/`up_proj` (expand MLP): minor quality loss; these layers have smooth
  activation distributions amenable to group-16 scaling.
- NVFP4 on attention projections (`q/k/v/o_proj`): eliminates the per-head scale underflow that
  all-NVFP4 Gemma suffers from — `o_proj` replaces the fused QKV path, so no multi-head `max()`
  across heads. Expect <0.3 PPL degradation vs BF16.
- FP8 dynamic on `down_proj`: these layers see large activation variance post-SiLU; dynamic
  per-token FP8 scales handle the range better than a calibrated global NVFP4 scale.
- Overall: estimated PPL increase ~0.4-0.6 over BF16 (vs ~1.0+ for all-NVFP4 with head underflow).

## Benchmark Plan

1. **Correctness baseline:** run `python fix_nvfp4_attn_to_bf16.py` on all-NVFP4 Gemma output;
   diff logits against this mixed recipe to confirm scale-underflow is resolved.
2. **Perplexity:** evaluate on WikiText-2 (stride-512) for BF16, all-NVFP4, and mixed recipe.
3. **Decode latency:** `uv run bench.py` at batch sizes B=1 and B=32, sequence length 2048,
   record `throughput_tflops`, `latency_us`, and `peak_vram_mb` in `results.tsv`.
4. **VRAM:** compare peak VRAM across BF16 (~52 GB), all-NVFP4 (~14 GB), mixed (~16 GB estimated).
5. **Scale-underflow check:** assert `abs(scale).min() > 1e-6` on all captured NVFP4 act scales
   post-calibration; flag any attention layer that still underflows.
