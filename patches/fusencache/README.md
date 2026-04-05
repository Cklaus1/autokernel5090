# FusenCache vLLM Patches

These are modified vLLM 0.19.0 files required for FusenCache integration.

## How to apply

Copy these files into your vLLM installation:

```bash
VLLM=$(python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))")
cp -r patches/fusencache/*.py "$VLLM/"
cp -r patches/fusencache/config/ "$VLLM/config/"
cp -r patches/fusencache/utils/ "$VLLM/utils/"
cp -r patches/fusencache/platforms/ "$VLLM/platforms/"
cp -r patches/fusencache/model_executor/ "$VLLM/model_executor/"
cp -r patches/fusencache/v1/ "$VLLM/v1/"
```

## Modified files

| File | Change |
|------|--------|
| `__init__.py` | Auto-import v4c patch on startup |
| `config/cache.py` | Added "fusen" to CacheDType |
| `utils/torch_utils.py` | Added "fusen" → torch.uint8 mapping |
| `v1/attention/backends/registry.py` | Added FUSENCACHE backend enum |
| `platforms/cuda.py` | Added FusenCache + TurboQuant routing |
| `model_executor/layers/attention/attention.py` | Added get_kv_cache_spec for fusen + sliding window |
| `v1/spec_decode/eagle.py` | Bypassed multimodal check + added Gemma4 image_token_id |

## Compatibility

- vLLM 0.19.0
- These patches will need updating for future vLLM versions
