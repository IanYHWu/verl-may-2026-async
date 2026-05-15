"""Dequantize a ModelOpt NVFP4 W4A16 HF checkpoint back to BF16.

For each quantized Linear, the checkpoint stores:
- `<m>.weight`         uint8, shape [out, in/2]  — packed FP4 (2 per byte)
- `<m>.weight_scale`   fp8 e4m3, shape [out, in/16] — per-block scales (block=16)
- `<m>.weight_scale_2` fp32 scalar — per-tensor global scale
- `<m>.input_scale`    fp32 scalar — runtime activation scale (DROPPED)
- (optional) k_scale / v_scale — KV cache scales (DROPPED)

Dequant: bf16_w = e2m1[unpack(packed)] * (weight_scale * weight_scale_2)
applied per-block along the last (in_features) axis.

Result is a plain BF16 HF checkpoint suitable as an init for verl QAT.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
BLOCK_SIZE = 16


def unpack_fp4(packed: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """packed: uint8 [..., in/2]  →  float32 [..., in]"""
    low = (packed & 0x0F).long()
    high = (packed >> 4).long()
    out_shape = (*packed.shape[:-1], packed.shape[-1] * 2)
    out = torch.empty(out_shape, dtype=torch.float32)
    out[..., 0::2] = lut[low]
    out[..., 1::2] = lut[high]
    return out


def dequant_one(packed: torch.Tensor, scale: torch.Tensor, gscale: torch.Tensor) -> torch.Tensor:
    """packed [out, in/2] uint8, scale [out, in/16] fp8, gscale scalar fp32 → bf16 [out, in]"""
    out_f, in_half = packed.shape
    in_features = in_half * 2
    assert scale.shape == (out_f, in_features // BLOCK_SIZE), (
        f"scale shape {scale.shape} mismatch for packed {packed.shape}"
    )

    unpacked = unpack_fp4(packed, E2M1)  # fp32 [out, in]
    per_block_scale = scale.to(torch.float32) * gscale.to(torch.float32)  # [out, in/16]
    deq = unpacked.view(out_f, -1, BLOCK_SIZE) * per_block_scale.unsqueeze(-1)
    return deq.reshape(out_f, in_features).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source NVFP4 model dir (single-file HF)")
    ap.add_argument("--dst", required=True, help="Destination BF16 model dir")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Pass through everything except the actual weight binary
    for p in src.iterdir():
        if p.is_dir() or p.name == "model.safetensors":
            continue
        shutil.copy2(p, dst / p.name)

    # Rewrite config.json to drop quantization_config
    cfg_path = dst / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.pop("quantization_config", None)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    # Drop hf_quant_config.json from dst (presence may confuse some loaders)
    (dst / "hf_quant_config.json").unlink(missing_ok=True)

    # Walk the safetensors and dequantize quantized weights, pass everything
    # else through (possibly cast to bf16 if it's an oddball dtype).
    sd = {}
    src_sft = src / "model.safetensors"
    with safe_open(src_sft, framework="pt") as f:
        keys = sorted(f.keys())
        # Map of base_name (no suffix) → {"weight": t, "scale": t, "gscale": t}
        groups: dict[str, dict[str, torch.Tensor]] = {}
        passthrough_keys = []
        for k in keys:
            if k.endswith(".weight_scale"):
                base = k[: -len(".weight_scale")]
                groups.setdefault(base, {})["scale"] = f.get_tensor(k)
            elif k.endswith(".weight_scale_2"):
                base = k[: -len(".weight_scale_2")]
                groups.setdefault(base, {})["gscale"] = f.get_tensor(k)
            elif k.endswith(".input_scale") or k.endswith(".k_scale") or k.endswith(".v_scale"):
                # Drop activation/KV scales — useless for BF16 init
                continue
            elif k.endswith(".weight"):
                passthrough_keys.append(k)
            else:
                passthrough_keys.append(k)

        # First pass: passthrough non-quantized weights as-is (cast to bf16 if needed)
        for k in passthrough_keys:
            t = f.get_tensor(k)
            if k.endswith(".weight"):
                base = k[: -len(".weight")]
                if base in groups and "scale" in groups[base]:
                    # Quantized — defer to dequant below
                    groups[base]["packed"] = t
                    continue
            # Plain weight (embed, norms, etc.) — preserve original dtype
            sd[k] = t

    # Second pass: dequantize each quantized linear
    n_quant = 0
    for base, parts in groups.items():
        if "packed" not in parts:
            # Group exists from a stray *_scale key but no packed weight — skip
            continue
        if "scale" not in parts or "gscale" not in parts:
            raise RuntimeError(f"incomplete quant group at {base}: {list(parts)}")
        sd[f"{base}.weight"] = dequant_one(parts["packed"], parts["scale"], parts["gscale"])
        n_quant += 1
        if n_quant % 50 == 0:
            print(f"  dequantized {n_quant} layers")

    print(f"dequantized {n_quant} layers, total tensors written {len(sd)}")
    save_file(sd, dst / "model.safetensors", metadata={"format": "pt"})
    print(f"wrote {dst}/model.safetensors")


if __name__ == "__main__":
    main()
