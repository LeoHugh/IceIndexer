"""用 forward hook 采集 DeepSeek-V3.2 DSA indexer 的 pre-RoPE q / k。

为什么能挂在 wq_b / k_norm 上：
  - Indexer._get_q_k_bf16() 里 query = wq_b(q_lora) 后 reshape 成 [T, n_heads, head_dim]，
    此时 RoPE 还没做（roPE 在 rotary_emb 那一步，且只作用于前 rope_head_dim 维）；
  - key = k_norm(wk(x)) 之后同样还没做 RoPE。
  所以这两个模块的 output 就是「全频带、pre-RoPE」的 q 和 k。

前提（缺一不可）：
  1. SGLANG_DISABLE_DSA_INDEXER_FUSION=1
     否则 wk / weights_proj / wq_b 会被折进融合 kernel，RoPE 在 CUDA kernel 内部完成，
     Python 侧拿不到 pre-RoPE 张量，hook 也不会被调用。
  2. --disable-cuda-graph
     CUDA Graph replay 不会执行 Python hook（只在 capture 时执行一次）。

环境变量：
  INDEXER_QK_OUT_DIR        必填，输出目录；不设则完全不生效
  INDEXER_QK_ONLY_PREFILL   默认 1，只采集 prefill（decode 每步都会算，量会爆炸）
  INDEXER_QK_LAYERS         可选，如 "0,10,20"；不设则采集该进程负责的全部层
  INDEXER_QK_MAX_STEPS      可选，每个 layer 最多采集多少步，默认 0 = 不限
  INDEXER_QK_CAST           可选 bf16(默认) / fp16 / fp32
  INDEXER_QK_ASYNC          用0
  INDEXER_QK_ASYNC_WORKERS  默认 2
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.util
import os
import sys

_TARGET_MODULES = (
    "sglang.srt.layers.attention.dsa.dsa_indexer",
    "sglang.srt.layers.attention.dsa.dsa_indexer_kpool",
)

_INSTALLED = False
_PATCHED_MODULES = set()
_EXECUTOR = None
_RANKS = {}


# --------------------------------------------------------------------------- utils
def _env(name, default=None):
    return os.environ.get(name, default)


def _int_env(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _first_tensor(obj):
    import torch

    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _ranks():
    if _RANKS:
        return _RANKS
    pp = tp = 0
    try:
        from sglang.srt.distributed import get_pp_group, get_tp_group

        pp = get_pp_group().rank_in_group
        tp = get_tp_group().rank_in_group
    except Exception:
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                pp = dist.get_rank()
        except Exception:
            pass
    _RANKS.update(pp=pp, tp=tp)
    return _RANKS


def _save(path, payload):
    import torch

    global _EXECUTOR
    if os.environ.get("INDEXER_QK_ASYNC", "1") == "1":
        if _EXECUTOR is None:
            from concurrent.futures import ThreadPoolExecutor

            _EXECUTOR = ThreadPoolExecutor(
                max_workers=_int_env("INDEXER_QK_ASYNC_WORKERS", 2)
            )
        _EXECUTOR.submit(torch.save, payload, path)
    else:
        torch.save(payload, path)


# --------------------------------------------------------------------------- dump
def _flush(indexer, q_raw, k_raw, positions, forward_batch):
    out_dir = _env("INDEXER_QK_OUT_DIR")
    if out_dir is None or k_raw is None or k_raw.shape[0] == 0:
        return

    if _env("INDEXER_QK_ONLY_PREFILL", "1") == "1" and forward_batch is not None:
        try:
            if forward_batch.forward_mode.is_decode_or_idle():
                return
        except Exception:
            pass

    want_layers = _env("INDEXER_QK_LAYERS")
    if want_layers:
        allowed = {s.strip() for s in want_layers.split(",")}
        if str(indexer.layer_id) not in allowed:
            return

    step = getattr(indexer, "_dsa_qk_step", 0)
    max_steps = _int_env("INDEXER_QK_MAX_STEPS", 0)
    if max_steps and step >= max_steps:
        return
    indexer._dsa_qk_step = step + 1

    import torch

    cast = _env("INDEXER_QK_CAST", "bf16")
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(cast, torch.bfloat16)

    ranks = _ranks()
    layer_dir = os.path.join(
        out_dir, f"pp{ranks['pp']:02d}_tp{ranks['tp']:02d}"
    )
    os.makedirs(layer_dir, exist_ok=True)

    n_heads = int(indexer.n_heads)
    head_dim = int(indexer.head_dim)

    q_cpu = None
    if q_raw is not None:
        # wq_b 的输出是 [T, n_heads * head_dim]，与 code 里的 rearrange 一致
        q_cpu = (
            q_raw.detach()
            .reshape(-1, n_heads, head_dim)
            .to(dtype=dtype, device="cpu")
        )
    k_cpu = k_raw.detach().reshape(-1, head_dim).to(dtype=dtype, device="cpu")

    payload = {
        "q_pre_rope": q_cpu,  # [T, n_heads, head_dim]
        "k_pre_rope": k_cpu,  # [T, head_dim]（indexer 的 k 是单头）
        "positions": positions.detach().cpu() if positions is not None else None,
        "layer_id": int(indexer.layer_id),
        "pp_rank": ranks["pp"],
        "tp_rank": ranks["tp"],
        "step": step,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "rope_head_dim": int(getattr(indexer, "rope_head_dim", 0)),
        "pid": os.getpid(),
    }
    path = os.path.join(layer_dir, f"l{indexer.layer_id:02d}_s{step:04d}.pt")
    _save(path, payload)


# --------------------------------------------------------------------------- attach
def _attach(indexer):
    if getattr(indexer, "_dsa_qk_attached", False):
        return
    indexer._dsa_qk_attached = True
    indexer._dsa_qk_step = 0

    if getattr(indexer, "use_dsa_indexer_fusion", False):
        print(
            "[dsa_qk_dump] WARNING: indexer fusion 已开启，wq_b/k_norm 不会走 Python，"
            "请设置 SGLANG_DISABLE_DSA_INDEXER_FUSION=1 并重启。",
            flush=True,
        )

    if not (hasattr(indexer, "wq_b") and hasattr(indexer, "k_norm")):
        print(
            f"[dsa_qk_dump] {type(indexer).__name__} 没有 wq_b/k_norm，跳过采集。",
            flush=True,
        )
        return

    state = {"q": None, "positions": None, "fb": None}
    indexer._dsa_qk_state = state

    def _pre_hook(module, args, kwargs):
        # 每次 forward 开头清空，防止上次残留的 q 被错配
        state["q"] = None
        state["positions"] = None
        state["fb"] = None
        try:
            import torch

            from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        except Exception:
            return
        for item in list(args) + list(kwargs.values()):
            if isinstance(item, ForwardBatch):
                state["fb"] = item
            elif (
                state["positions"] is None
                and torch.is_tensor(item)
                and item.dtype in (torch.int64, torch.int32)
            ):
                state["positions"] = item

    def _q_hook(module, args, output):
        tensor = _first_tensor(output)
        if tensor is not None:
            state["q"] = tensor

    def _k_hook(module, args, output):
        key = _first_tensor(output)
        query = state["q"]
        state["q"] = None
        try:
            _flush(indexer, query, key, state["positions"], state["fb"])
        except Exception as exc:  # 采集失败不能影响推理
            print(f"[dsa_qk_dump] flush failed: {exc!r}", flush=True)

    indexer.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    indexer.wq_b.register_forward_hook(_q_hook)
    indexer.k_norm.register_forward_hook(_k_hook)
    print(
        f"[dsa_qk_dump] hooked layer_id={indexer.layer_id} "
        f"n_heads={indexer.n_heads} head_dim={indexer.head_dim} "
        f"rope_head_dim={getattr(indexer, 'rope_head_dim', None)}",
        flush=True,
    )


def _patch(module):
    for cls_name in ("Indexer", "IndexerKPool"):
        cls = getattr(module, cls_name, None)
        if cls is None or getattr(cls, "_dsa_qk_patched", False):
            continue
        cls._dsa_qk_patched = True
        orig_init = cls.__init__

        @functools.wraps(orig_init)
        def __init__(self, *args, __orig=orig_init, **kwargs):
            __orig(self, *args, **kwargs)
            _attach(self)

        cls.__init__ = __init__


# --------------------------------------------------------------------------- import hook
class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            _patch(module)
        except Exception as exc:
            print(f"[dsa_qk_dump] patch failed: {exc!r}", flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGET_MODULES or fullname in _PATCHED_MODULES:
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        _PATCHED_MODULES.add(fullname)
        spec.loader = _Loader(spec.loader)
        return spec


def install():
    """在每个 sglang 进程里调用一次（sitecustomize.py 会自动做）。"""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    sys.meta_path.insert(0, _Finder())
    # 若模块已被提前 import，直接补打
    for name in _TARGET_MODULES:
        module = sys.modules.get(name)
        if module is not None:
            try:
                _patch(module)
            except Exception as exc:
                print(f"[dsa_qk_dump] patch failed: {exc!r}", flush=True)
