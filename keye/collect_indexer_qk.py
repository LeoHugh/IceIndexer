"""
采集 Keye-VL-2.0 每层 indexer 在 norm 之后、RoPE 之前的 q / k，以及 head 权重 w。

不修改模型源码，全部通过 hook 完成：
  - sa_indexer.q_norm 的输出  -> q  [L, 16, 64]   （norm 后、RoPE 前）
  - sa_indexer.k_norm 的输出  -> k  [L, 64]       （MQA，只有 1 个 kv head）
  - sa_indexer.q_proj 输出的最后 16 维 -> w [L, 16]
  - 可选 --save_prenorm：q_proj / k_proj 的原始输出（norm 前）
长序列用分块 prefill（通过 KV cache 逐块喂入），显存只和 chunk × 序列长度 成正比。
第一条序列结束后，会用存下来的 q/k/w 自己重算 indexer 分数和 top-k，与模型实际选出的 top-k 对比，
用来确认取值位置、RoPE 配对方式和频率都理解正确。

用法:
CUDA_VISIBLE_DEVICES=4,5,6,7 python collect_indexer_qk.py \
    --data /home/v-lihucheng/lbv2_50k.jsonl \
    --out indexer_qk \
    --chunk 2048 \
    --max_len 51200
数据格式（jsonl，每行一条）:
  {"id": "aime_001", "input_ids": [...]}
  或 {"id": "aime_001", "text": "已经套好 chat template 的完整文本"}
读取:
  from safetensors import safe_open
  with safe_open("indexer_qk/aime_001.safetensors", "pt") as f:
      q = f.get_tensor("layer10.q")   # [L, 16, 64]
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
p.add_argument("--out", default="indexer_qk")
p.add_argument("--chunk", type=int, default=1024, help="分块 prefill 的块大小；显存不够就调小（如 512）")
p.add_argument("--max_len", type=int, default=32768)
p.add_argument("--save_prenorm", action="store_true", help="同时保存 norm 之前的原始投影")
p.add_argument("--verify_layers", default="0,24", help="第一条序列结束后做 top-k 对照验证的层")
p.add_argument("--no_verify", action="store_true")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
    max_memory={i: "74GiB" for i in range(torch.cuda.device_count())},  # 跟随 CUDA_VISIBLE_DEVICES 的卡数
    attn_implementation="eager",  # 必须是 eager，sdpa 下因果 mask 会被跳过
)
model.eval()
assert model.model.config._attn_implementation == "eager", "必须用 eager，否则 prefill 是双向 attention"
dev = model.device

layers = model.model.layers
NL = len(layers)
ix0 = layers[0].self_attn.sa_indexer
H, D, QD, TOPK = ix0.num_heads, ix0.head_dim, ix0.total_q_dim, ix0.topk
assert ix0.num_kv_heads == 1, "这份脚本按 MQA（1 个 kv head）写的"
print(f"层数={NL}  indexer heads={H}  head_dim={D}  topk={TOPK}  "
      f"rope inv_freq 个数={ix0.rotary_emb.inv_freq.numel()}  mrope 分段={ix0.indexer_mrope_section}")

# ------------------------------------------------------------------ hook：把每块的结果直接写进 CPU 缓冲区
state = {"start": 0, "store": None}


def put(i, name, t):
    s = state["start"]
    state["store"][i][name][s:s + t.shape[0]].copy_(t)


def h_qnorm(i):
    def h(mod, a, out):
        put(i, "q", out[0])                      # [n, H, D]
    return h


def h_knorm(i):
    def h(mod, a, out):
        put(i, "k", out[0, :, 0])                # [n, 1, D] -> [n, D]
    return h


def h_qproj(i):
    def h(mod, a, out):
        put(i, "w", out[0, :, QD:])              # [n, H]
        if args.save_prenorm:
            put(i, "q_pre", out[0, :, :QD].reshape(-1, H, D))
    return h


def h_kproj(i):
    def h(mod, a, out):
        put(i, "k_pre", out[0].reshape(-1, D))   # [n, D]
    return h


for i, layer in enumerate(layers):
    ix = layer.self_attn.sa_indexer
    ix.q_norm.register_forward_hook(h_qnorm(i))
    ix.k_norm.register_forward_hook(h_knorm(i))
    ix.q_proj.register_forward_hook(h_qproj(i))
    if args.save_prenorm:
        ix.k_proj.register_forward_hook(h_kproj(i))

# 位置检查：记录第 0 层 indexer rotary 实际收到的位置
pos_seen = []
ix0.rotary_emb.register_forward_pre_hook(lambda mod, a: pos_seen.append(a[1][0, 0].detach().cpu()))

# ------------------------------------------------------------------ 记录模型实际选出的 top-k，用于验证
VERIFY = set() if args.no_verify else {int(x) for x in args.verify_layers.split(",") if x != ""}
rec = {"on": False}
attn_cls = type(layers[0].self_attn)
_orig_build = attn_cls._build_topk_mask


def _build_wrapped(self, topk_indices, q_len, kv_len, *a, **kw):
    if rec["on"] and self.layer_idx in VERIFY:
        rec[self.layer_idx] = (kv_len - q_len, topk_indices[0].cpu())  # 只保留最后一块
    return _orig_build(self, topk_indices, q_len, kv_len, *a, **kw)


attn_cls._build_topk_mask = _build_wrapped


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def verify(store, L):
    for li in sorted(VERIFY):
        if li not in rec:
            continue
        start, idx = rec[li]
        ix = layers[li].self_attn.sa_indexer
        inv = ix.rotary_emb.inv_freq.float().cpu()                  # [D/2]
        q, k, w = (store[li][n].float() for n in ("q", "k", "w"))
        fr = torch.arange(L, dtype=torch.float32)[:, None] * inv[None]
        cos = torch.cat([fr.cos(), fr.cos()], -1)                    # 纯文本时 mRoPE 三个分量相同，等价于普通 RoPE
        sin = torch.cat([fr.sin(), fr.sin()], -1)
        k_rot = k * cos + rotate_half(k) * sin
        rows = torch.linspace(0, idx.shape[0] - 1, steps=min(8, idx.shape[0])).long().tolist()
        ovs = []
        for r in rows:
            t = start + r
            qt = q[t] * cos[t] + rotate_half(q[t]) * sin[t]          # [H, D]
            sc = (qt @ k_rot[: t + 1].T) * ix.scale                  # Hadamard 是正交变换，不改变点积，这里省略
            I = (w[t][:, None] * sc.relu()).sum(0)
            kk = min(TOPK, t + 1)
            mine = set(I.topk(kk).indices.tolist())
            theirs = set(idx[r, :kk].tolist())
            ovs.append(len(mine & theirs) / kk)
        print(f"  [验证] layer {li}: query 位置 {start}~{start + idx.shape[0] - 1} 中抽 {len(rows)} 行，"
              f"自己重算的 top-k 与模型实际 top-k 的重合率: {[round(o, 3) for o in ovs]}")
        if start + 1 <= TOPK:
            print("    注意：这些 query 的历史长度不超过 topk，top-k 等于全选，重合率必然为 1，不能说明问题；"
                  "请用长度明显大于 topk 的序列验证")


def alloc(L):
    st = []
    for _ in range(NL):
        d = {"q": torch.empty(L, H, D, dtype=torch.bfloat16),
             "k": torch.empty(L, D, dtype=torch.bfloat16),
             "w": torch.empty(L, H, dtype=torch.bfloat16)}
        if args.save_prenorm:
            d["q_pre"] = torch.empty(L, H, D, dtype=torch.bfloat16)
            d["k_pre"] = torch.empty(L, D, dtype=torch.bfloat16)
        st.append(d)
    return st


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
first = True
for sid, ids in load_items(args.data):
    out_path = os.path.join(args.out, f"{sid}.safetensors")
    if os.path.exists(out_path):
        print(f"[跳过] {sid} 已存在")
        continue
    L = len(ids)
    store = alloc(L)
    state["store"] = store
    x = torch.tensor(ids, device=dev)[None]
    past = None
    pos_seen.clear()
    rec_on = first and bool(VERIFY)
    rec["on"] = rec_on
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
    rec["on"] = False

    got = torch.cat(pos_seen)
    assert torch.equal(got, torch.arange(L)), f"indexer 收到的位置与预期不一致: {got[:5]} ... {got[-5:]}"

    tensors = {f"layer{i}.{name}": t for i, d in enumerate(store) for name, t in d.items()}
    tensors["input_ids"] = torch.tensor(ids, dtype=torch.int32)
    save_file(tensors, out_path, metadata={
        "id": sid, "length": str(L), "chunk": str(args.chunk),
        "stage": "q/k: after norm, before RoPE; w: raw",
        "rope_pairing": "rotate_half: complex band f = (x[f], x[f + D/2])",
    })
    gb = sum(t.numel() * t.element_size() for t in tensors.values()) / 1e9
    print(f"[完成] {sid}: L={L}  用时 {time.time() - t0:.1f}s  文件 {gb:.2f} GB -> {out_path}")

    if rec_on:
        verify(store, L)
    first = False
    del past, store, tensors
    torch.cuda.empty_cache()