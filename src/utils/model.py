"""
model.py — vLLM 엔진 로드 및 환경 호환성 패치.
"""
import importlib
import os
from typing import Optional

import torch
from transformers import PretrainedConfig

from ..config import (
    MAX_NEW_TOKENS,
    MAX_PIXELS,
    MIN_PIXELS,
    MODEL_NAME,
    REPETITION_PENALTY,
    STAGE1_MAX_FRAMES,
    STAGE1_TEMPERATURE,
    TEMPERATURE,
    TOP_P,
    VLLM_GPU_MEMORY_UTIL,
    VLLM_MAX_MODEL_LEN,
)


def patch_transformers_rope_validation_compat() -> None:
    original = getattr(PretrainedConfig, "_check_received_keys", None)
    if original is None or getattr(original, "_qwen35_vllm_compat", False):
        return

    def patched(rope_type, received_keys, required_keys, optional_keys=None, ignore_keys=None):
        if ignore_keys is not None and not isinstance(ignore_keys, set):
            ignore_keys = set(ignore_keys)
        return original(rope_type, received_keys, required_keys,
                        optional_keys=optional_keys, ignore_keys=ignore_keys)

    patched._qwen35_vllm_compat = True
    PretrainedConfig._check_received_keys = staticmethod(patched)


def infer_tensor_parallel_size() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return max(1, len([t for t in visible.split(",") if t.strip()]))
    return max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1


def load_vllm_model(model_name: Optional[str] = None):
    patch_transformers_rope_validation_compat()
    try:
        vllm_mod       = importlib.import_module("vllm")
        LLM            = getattr(vllm_mod, "LLM")
        SamplingParams = getattr(vllm_mod, "SamplingParams")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("vllm이 설치되지 않았습니다. pip install vllm") from exc

    effective_model = model_name or MODEL_NAME
    tp = infer_tensor_parallel_size()

    print(f"  -> vLLM 로드: model={effective_model}, tp={tp}, max_model_len={VLLM_MAX_MODEL_LEN}")
    model = LLM(
        model=effective_model, tokenizer=effective_model,
        trust_remote_code=True, tensor_parallel_size=tp, dtype="bfloat16",
        max_model_len=VLLM_MAX_MODEL_LEN, gpu_memory_utilization=VLLM_GPU_MEMORY_UTIL,
        enforce_eager=True, disable_custom_all_reduce=True,
        limit_mm_per_prompt={"image": STAGE1_MAX_FRAMES},
        mm_processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
        max_num_seqs=1,
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
        top_p=TOP_P, repetition_penalty=REPETITION_PENALTY,
    )
    stage1_sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, temperature=STAGE1_TEMPERATURE,
        top_p=TOP_P, repetition_penalty=REPETITION_PENALTY,
    )
    return model, sampling_params, stage1_sampling_params
