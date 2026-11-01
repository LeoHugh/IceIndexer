"""
记录 Keye-VL-2.0 每层 indexer 选 top-k 时，每个 query token 选中了哪些 KV 位置。

思路与 collect_indexer_qk.py 一致：不改模型源码，长序列分块 prefill。
区别是这次不采 q/k/w，而是直接截获模型真实算出的 topk_indices：
包装 KeyeTopKMaskAttention._build_topk_mask。prefill 分支里 indexer 是对
「从头开始的完整 KV」打分，所以截到的下标就是全序列的绝对位置，跨 chunk 连续。

每条序列保存一个 safetensors:
  layer{i}.topk_idx  [L, topk]  int32
      第 t 行 = 位置 t 的 query 选中的 KV 位置，按 indexer 分数从高到低排列；
      因果约束下位置 t 只能看到 0..t，所以真实选择只有前 min(topk, t+1) 列，
      其余列（因果外的填充位、未选中的位置）一律置 -1。
  input_ids  [L]  int32   配合 topk_idx 可把位置映射回具体 token

注意:
  - 只覆盖 prefill 阶段（本脚本与 collect_indexer_qk.py 一样只做 prefill 不 decode）。
  - 体积 = 层数 × L × topk × 4 字节。L=50k、全部 48 层时每条约 19.7 GB，
    不需要全部层时用 --layers 0,12,24,47 这样的子集。

用法:
CUDA_VISIBLE_DEVICES=2,3 python collect_indexer_topk.py \
    --data /home/v-lihucheng/lbv2_50k.jsonl \
    --out indexer_topk \
    --chunk 2048 \
    --max_len 51200 \
    --layers all

读取:
  from safetensors import safe_open
  with safe_open("indexer_topk/<id>.safetensors", "pt") as f:
      idx = f.get_tensor("layer24.topk_idx")   # [L, 2048] int32，-1 = 无效
"""
import argparse
import json
import os
import time

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="/home/v-lihucheng/keye-vl-30b")
p.add_argument("--data", required=True)
p.add_argument("--out", default="indexer_topk")
p.add_argument("--chunk", type=int, default=1024, help="分块 prefill 的块大小；显存不够就调小（如 512）")
p.add_argument("--max_len", type=int, default=32768)
p.add_argument("--layers", default="all", help="要记录的层：all 或逗号分隔，如 0,12,24,47")
p.add_argument("--gpu_mem", default="74GiB", help="每张卡最多用多少显存；和别人共享显卡时调低，如 46GiB")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
    max_memory={i: args.gpu_mem for i in range(torch.cuda.device_count())},  # 跟随 CUDA_VISIBLE_DEVICES 的卡数
    attn_implementation="eager",  # 必须是 eager，sdpa 下因果 mask 会被跳过
)
model.eval()
assert model.model.config._attn_implementation == "eager", "必须用 eager，否则 prefill 是双向 attention"
dev = model.device

layers = model.model.layers
NL = len(layers)
TOPK = layers[0].self_attn.sa_indexer.topk
LAYERS = set(range(NL)) if args.layers == "all" else {int(x) for x in args.layers.split(",") if x != ""}
assert LAYERS and all(0 <= i < NL for i in LAYERS), f"层号要在 0~{NL - 1} 内"
est_gb = len(LAYERS) * min(args.max_len, 10 ** 9) * TOPK * 4 / 1e9
print(f"层数={NL}  topk={TOPK}  记录层={sorted(LAYERS)}  "
      f"按 max_len={args.max_len} 估算每条序列约 {est_gb:.1f} GB")

# ------------------------------------------------------------------ 截获模型真实选出的 top-k 下标
state = {"start": 0, "store": None}

attn_cls = type(layers[0].self_attn)
_orig_build = attn_cls._build_topk_mask


def _build_wrapped(self, topk_indices, q_len, kv_len, *a, **kw):
    st = state["store"]
    if st is not None and self.layer_idx in LAYERS:
        s = state["start"]
        w = topk_indices.shape[-1]                                  # 本块实际个数 = min(topk, 当前 kv 长度)
        st[self.layer_idx][s:s + q_len, :w].copy_(topk_indices[0])  # [q_len, w]，绝对位置
    return _orig_build(self, topk_indices, q_len, kv_len, *a, **kw)


attn_cls._build_topk_mask = _build_wrapped

# 位置检查：确认分块喂入时 indexer 收到的位置是 0..L-1 连续
pos_seen = []
layers[0].self_attn.sa_indexer.rotary_emb.register_forward_pre_hook(
    lambda mod, a: pos_seen.append(a[1][0, 0].detach().cpu())
)


def load_items(path):
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            ids = ex["input_ids"] if "input_ids" in ex else tok(ex["text"], add_special_tokens=False).input_ids
            yield str(ex.get("id", f"seq{n:05d}")), ids[: args.max_len]


# ------------------------------------------------------------------ 主循环
for sid, ids in load_items(args.data):
    out_path = os.path.join(args.out, f"{sid}.safetensors")
    if os.path.exists(out_path):
        print(f"[跳过] {sid} 已存在")
        continue
    L = len(ids)
    store = {i: torch.full((L, TOPK), -1, dtype=torch.int32) for i in sorted(LAYERS)}
    state["store"] = store
    x = torch.tensor(ids, device=dev)[None]
    past = None
    pos_seen.clear()
    t0 = time.time()

    with torch.inference_mode():
        for s in range(0, L, args.chunk):
            e = min(L, s + args.chunk)
            state["start"] = s
            pos = torch.arange(s, e, device=dev)
            out = model(
                input_ids=x[:, s:e],
                attention_mask=torch.ones(1, e, dtype=torch.long, device=dev),
                position_ids=pos.view(1, 1, -1).expand(3, 1, -1),   # 显式传位置，避免分块时位置从 0 重新开始
                cache_position=pos,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            del out

    got = torch.cat(pos_seen)
    assert torch.equal(got, torch.arange(L)), f"indexer 收到的位置与预期不一致: {got[:5]} ... {got[-5:]}"

    # 清洗：位置 t 因果上只能看到 0..t，真实选择只有前 min(topk, t+1) 个，尾部全部置 -1
    # （topk 按分数降序返回，被因果 mask 掉的未来位置分数是 finfo.min，一定排在有效选择后面）
    n_valid = torch.clamp(torch.arange(L) + 1, max=TOPK)
    dirty = torch.arange(TOPK)[None, :] >= n_valid[:, None]          # [L, topk]
    fut = torch.arange(L)[:, None]
    for i, arr in store.items():
        arr[dirty] = -1
        assert not ((arr != -1) & (arr > fut)).any(), f"layer {i} 出现了选到未来位置的 top-k"

    tensors = {f"layer{i}.topk_idx": arr for i, arr in store.items()}
    tensors["input_ids"] = torch.tensor(ids, dtype=torch.int32)
    save_file(tensors, out_path, metadata={
        "id": sid, "length": str(L), "chunk": str(args.chunk), "topk": str(TOPK),
        "semantics": "row t = KV positions selected by the query at position t, sorted by indexer score "
                     "descending; only the first min(topk, t+1) entries are real selections, the rest are -1",
    })
    gb = sum(t.numel() * t.element_size() for t in tensors.values()) / 1e9
    print(f"[完成] {sid}: L={L}  层数={len(LAYERS)}  用时 {time.time() - t0:.1f}s  "
          f"文件 {gb:.2f} GB -> {out_path}")

    del past, store, tensors
    torch.cuda.empty_cache()
