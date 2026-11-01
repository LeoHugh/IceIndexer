"""
评测把高频带换成离线中心之后，indexer 选出的 top-k 损失了什么：
  1. 丢失 / 新增 key 的距离 Δ 分布（按 2 的幂分桶），以及丢失 key 携带的主干 attention 质量
  2. indexer 打分质量召回率：原始 indexer 的 softmax(I) 分布中，被近似 top-k 覆盖的质量（相对原始 top-k）
  3. 主干 attention 质量召回率：dense attention 分布中，被选中的 2048 个 key 覆盖的质量
     （4 个 GQA 组分别计算后取平均，同时报告最差组；oracle = 直接按 attention 取 top-k 的上界）

需要重新跑模型：主干 attention 的 q / K 之前没有采集。每个文件做一次分块 prefill（eager），
用 hook 抓 indexer 的 q/k/w 和主干的 q（只在抽样的 query 位置），prefill 结束后直接从 KV cache 读主干 K。
不改模型源码。每条 50K 序列约 2.5~3 分钟。

对比的方案（中心都用留一：去掉当前文件后其余文件的中心）:
  full       原始 indexer（参照）
  drop       删掉 B 频段，不加偏置
  table_loo  删掉 B 频段 + 偏置表（q、k 都用中心）
  kmean_loo  只把 k 的 B 频段换成中心，q 保持动态
  oracle     按 dense attention 质量直接取 top-k（上界）

用法:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_attn_recall.py --data lbv2_50k.jsonl --calib calib --bands 0-15 
"""
import argparse
import collections
import csv
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM


def parse_bands(s, F):
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    out = sorted(set(out))
    assert out and all(0 <= f < F for f in out), f"频段越界: {out}"
    return out


p = argparse.ArgumentParser()
p.add_argument("--model", default="/home/v-lihucheng/keye-vl-30b")
p.add_argument("--data", required=True, help="prep_longbench_v2.py 输出的 jsonl")
p.add_argument("--calib", default="calib")
p.add_argument("--bands", default=None, help="默认用第一步的 --bands")
p.add_argument("--out", default="eval_attn")
p.add_argument("--n_files", type=int, default=0, help="0 = 全部")
p.add_argument("--chunk", type=int, default=2048)
p.add_argument("--max_len", type=int, default=51200)
p.add_argument("--n_queries", type=int, default=64)
p.add_argument("--min_pos", type=int, default=4096)
p.add_argument("--topk", type=int, default=2048)
p.add_argument("--qbatch", type=int, default=16)
p.add_argument("--hybrid_tol", type=float, default=0.005,
               help="混合方案：table 的 attention 召回比 full 低不超过这个值的层用 table，其余层用 kmean")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--sampling", default="stratified", choices=["stratified", "random"],
               help="stratified: 把 [min_pos, L) 等分成 n_queries 段，每段抽一个，保证各个位置都有覆盖")
args = p.parse_args()
assert args.min_pos >= args.topk, "--min_pos 必须 >= --topk"
os.makedirs(args.out, exist_ok=True)
K = args.topk

# ---------------------------------------------------------------- 离线中心
C = np.load(os.path.join(args.calib, "centers.npz"))
P = np.load(os.path.join(args.calib, "per_file_stats.npz"))
q_bar, k_bar = C["q_bar"], C["k_bar"]
DI = int(C["head_dim"]); FI = DI // 2
B = parse_bands(args.bands, FI) if args.bands else [int(x) for x in C["default_bands"]]
calib_names = [str(x) for x in P["files"]]
Sq_all, Sk_all, ns = P["Sq"], P["Sk"], P["ns"]
Sq_tot, Sk_tot, n_tot = Sq_all.sum(0), Sk_all.sum(0), ns.sum()

# ---------------------------------------------------------------- 模型
model = AutoModelForCausalLM.from_pretrained(
    args.model, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
    max_memory={0: "62GiB", 1: "62GiB", 2: "62GiB", 3: "62GiB"}, attn_implementation="eager",
)
model.eval()
assert model.model.config._attn_implementation == "eager", "必须用 eager，否则 prefill 是双向 attention"
dev0 = model.device
layers = model.model.layers
NL = len(layers)
ix0, at0 = layers[0].self_attn.sa_indexer, layers[0].self_attn
HI, QD, scaleI = ix0.num_heads, ix0.total_q_dim, ix0.scale
assert ix0.head_dim == DI and ix0.num_kv_heads == 1
omI_np = ix0.rotary_emb.inv_freq.double().cpu().numpy()
assert np.allclose(omI_np, C["omega"], rtol=1e-5), "indexer 的 inv_freq 和第一步用的 omega 不一致"
HM, DM, G = at0.num_heads, at0.head_dim, at0.num_key_value_heads
FM = DM // 2
rot_main = model.model.rotary_emb
omM_np = rot_main.inv_freq.double().cpu().numpy()
att_scaling = float(getattr(rot_main, "attention_scaling", 1.0))
assert omM_np.size == FM, f"主干 inv_freq 长度 {omM_np.size} != head_dim/2"
print(f"{NL} 层 | indexer: {HI} heads × {DI} | 主干: {HM} heads, {G} kv heads × {DM} | 替代频段 B={B}")

# ---------------------------------------------------------------- hook：只在抽样位置抓 q / w，indexer k 全量抓
BUF = {}
ST = {"s": 0, "loc": None, "dst": None}


def grab(buf, o):  # o: [1, n, ...]
    if ST["dst"].numel():
        buf[ST["dst"]] = o[0, ST["loc"].to(o.device)].to("cpu", buf.dtype)


def h_ixq(l):
    return lambda m, a, o: grab(BUF["qI"][l], o)                    # [1,n,HI,DI]


def h_ixk(l):
    def h(m, a, o):
        s = ST["s"]
        BUF["kI"][l][s:s + o.shape[1]].copy_(o[0, :, 0])            # [1,n,1,DI]
    return h


def h_ixw(l):
    return lambda m, a, o: grab(BUF["wI"][l], o[..., QD:])          # [1,n,QD+HI]


def h_mq(l):
    return lambda m, a, o: grab(BUF["qM"][l], o)                    # [1,n,HM,DM]，主干 q（norm 后、RoPE 前）


def h_mk(l):
    return lambda m, a, o: grab(BUF["kM"][l], o)                    # [1,n,G,DM]，用来自检 RoPE


for l, layer in enumerate(layers):
    ix = layer.self_attn.sa_indexer
    ix.q_norm.register_forward_hook(h_ixq(l))
    ix.k_norm.register_forward_hook(h_ixk(l))
    ix.q_proj.register_forward_hook(h_ixw(l))
    layer.self_attn.q_norm.register_forward_hook(h_mq(l))
    layer.self_attn.k_norm.register_forward_hook(h_mk(l))


def get_keys(past, l):
    return past.layers[l].keys if hasattr(past, "layers") else past.key_cache[l]


def to_z(x, F):
    return torch.complex(x[..., :F].contiguous(), x[..., F:].contiguous())


def polar(angle):
    return torch.polar(torch.ones_like(angle), angle)


# ---------------------------------------------------------------- Δ 分桶：0, [1,2), [2,4), ..., [2^15, 2^16)
EDGES = [2 ** i for i in range(17)]
NB = len(EDGES) + 1
BIN_LABELS = ["0"] + [f"[{2 ** (i - 1)},{2 ** i})" for i in range(1, NB)]
VARS = ["drop", "table_loo", "kmean_loo"]
hist = {
    "ref_cnt": np.zeros((NL, NB)), "ref_mass": np.zeros((NL, NB)),
    **{f"{v}_{kind}": np.zeros((NL, NB)) for v in VARS for kind in ("lost_cnt", "lost_mass", "added_cnt")},
    **{f"{v}_sink_lost": np.zeros(NL) for v in VARS},
}
n_query_total = 0
rows = []
k_rope_err = []
ALLV = ["full"] + VARS + ["oracle"]
METRICS = ("overlap", "idx", "attn", "attn_min")
PQ = {}   # 逐 query 的结果: sid -> (qpos, L, array[NL, len(ALLV), len(METRICS), nq])


@torch.inference_mode()
def eval_layer(l, Kc, qpos, L, cq, ck, fname, pq):
    d = Kc.device
    Km = Kc[0].float()                                             # [G,L,DM]，RoPE 之后
    t_all = torch.tensor(qpos, device=d)
    j = torch.arange(L, device=d)
    omI = torch.tensor(omI_np, device=d)
    omM = torch.tensor(omM_np, device=d)
    rotI = polar(torch.arange(L, device=d, dtype=torch.float64)[:, None] * omI[None]).to(torch.complex64)

    zk = to_z(BUF["kI"][l].to(d).float(), FI) * rotI
    kIr = torch.cat([zk.real, zk.imag], -1)                         # [L,DI]
    zq = to_z(BUF["qI"][l].to(d).float(), FI)                      # [n,HI,FI] RoPE 前
    zqr = zq * rotI[t_all][:, None]
    qIr = torch.cat([zqr.real, zqr.imag], -1)
    w = BUF["wI"][l].to(d).float()                                 # [n,HI]

    rotM = polar(t_all[:, None].double() * omM[None]).to(torch.complex64)[:, None]   # [n,1,FM]
    zqm = to_z(BUF["qM"][l].to(d).float(), FM) * rotM
    qMr = torch.cat([zqm.real, zqm.imag], -1) * att_scaling        # [n,HM,DM]
    zkm = to_z(BUF["kM"][l].to(d).float(), FM) * rotM              # 自检：自己算的 RoPE 应该和 cache 里的 K 一致
    kMr = torch.cat([zkm.real, zkm.imag], -1) * att_scaling
    kc = Km[:, t_all].permute(1, 0, 2)
    k_rope_err.append(((kMr - kc).norm() / kc.norm()).item())

    m_lo = torch.ones(DI, device=d)
    m_lo[B] = 0
    m_lo[[b + FI for b in B]] = 0
    omB = omI[B]
    A = torch.from_numpy(cq[l][:, B] * np.conj(ck[l][B])[None]).to(d)          # [HI,|B|]
    T = (A @ polar(omB[:, None] * torch.arange(L, device=d, dtype=torch.float64)[None])).real.float() * scaleI  # [HI,L]
    kb = torch.from_numpy(ck[l][B]).to(d).to(torch.complex64)
    edges = torch.tensor(EDGES, device=d)

    acc = collections.defaultdict(list)
    for s in range(0, len(qpos), args.qbatch):
        tb = t_all[s:s + args.qbatch]
        bsz = len(tb)
        qb_, wb = qIr[s:s + args.qbatch], w[s:s + args.qbatch]
        causal = j[None] > tb[:, None]
        dlt = (tb[:, None] - j[None]).clamp(min=0)
        bins = torch.bucketize(dlt, edges, right=True)             # [b,L]

        S_full = torch.einsum("bhd,ld->bhl", qb_, kIr) * scaleI
        S_lo = torch.einsum("bhd,ld->bhl", qb_ * m_lo, kIr) * scaleI

        def score(S):
            return (wb[:, :, None] * S.clamp(min=0)).sum(1).masked_fill(causal, float("-inf"))

        I_ref = score(S_full)
        sel = {"full": I_ref.topk(K, dim=-1).indices}
        pI = torch.softmax(I_ref, -1)                              # indexer 训练时对齐的就是 softmax(I)

        sel["drop"] = score(S_lo).topk(K, dim=-1).indices
        sel["table_loo"] = score(S_lo + T[:, dlt].permute(1, 0, 2)).topk(K, dim=-1).indices
        a_ = zq[s:s + args.qbatch][:, :, B] * kb.conj()[None, None]
        Eb = polar(omB[None, :, None] * dlt[:, None, :].double()).to(torch.complex64)
        sel["kmean_loo"] = score(S_lo + torch.einsum("bhf,bfl->bhl", a_, Eb).real * scaleI).topk(K, dim=-1).indices
        del S_full, S_lo, Eb

        # 主干 dense attention：每个 GQA 组内 8 个 head 取平均
        qg = qMr[s:s + args.qbatch].view(bsz, G, HM // G, DM)      # head h 对应 kv head h // 8（repeat_kv）
        att_g = torch.einsum("bgkd,gld->bgkl", qg, Km) / DM ** 0.5
        att_g = torch.softmax(att_g.masked_fill(causal[:, None, None], float("-inf")), -1).mean(2)   # [b,G,L]
        pA = att_g.mean(1)                                         # [b,L]
        sel["oracle"] = pA.topk(K, dim=-1).indices

        def attn_mass(ix_):                                        # [b,K] -> [b,G]
            return att_g.gather(2, ix_[:, None, :].expand(-1, G, -1)).sum(-1)

        ref_mask = torch.zeros_like(causal).scatter_(1, sel["full"], True)
        for v, ix_ in sel.items():
            am = attn_mass(ix_)
            acc[(v, "attn")].append(am.mean(1))
            acc[(v, "attn_min")].append(am.min(1).values)
            acc[(v, "idx")].append(pI.gather(1, ix_).sum(1))
            acc[(v, "overlap")].append(ref_mask.gather(1, ix_).float().mean(1))

        hist["ref_cnt"][l] += torch.bincount(bins[ref_mask], minlength=NB).cpu().numpy()
        hist["ref_mass"][l] += torch.zeros(NB, device=d).index_add_(0, bins[ref_mask], pA[ref_mask]).cpu().numpy()
        for v in VARS:
            vm = torch.zeros_like(causal).scatter_(1, sel[v], True)
            lost, added = ref_mask & ~vm, vm & ~ref_mask
            hist[f"{v}_lost_cnt"][l] += torch.bincount(bins[lost], minlength=NB).cpu().numpy()
            hist[f"{v}_lost_mass"][l] += torch.zeros(NB, device=d).index_add_(0, bins[lost], pA[lost]).cpu().numpy()
            hist[f"{v}_added_cnt"][l] += torch.bincount(bins[added], minlength=NB).cpu().numpy()
            hist[f"{v}_sink_lost"][l] += lost[:, :4].sum().item()
        del att_g, pA, pI, I_ref

    for vi, v in enumerate(ALLV):
        rows.append([fname, l, v] + [torch.cat(acc[(v, m)]).mean().item() for m in METRICS])
        for mi, m in enumerate(METRICS):
            pq[l, vi, mi] = torch.cat(acc[(v, m)]).float().cpu().numpy()


# ---------------------------------------------------------------- 主循环
items = []
with open(args.data, encoding="utf-8") as fh:
    for line in fh:
        if line.strip():
            ex = json.loads(line)
            items.append((str(ex["id"]), ex["input_ids"][: args.max_len]))
if args.n_files > 0:
    items = items[: args.n_files]

for fi, (sid, ids) in enumerate(items):
    t0 = time.time()
    L = len(ids)
    if L <= args.min_pos:
        print(f"[跳过] {sid}: L={L}")
        continue
    if sid in calib_names:
        a = calib_names.index(sid)
        cq = (Sq_tot - Sq_all[a]) / (n_tot - ns[a]); ck = (Sk_tot - Sk_all[a]) / (n_tot - ns[a])
    else:
        cq, ck = q_bar, k_bar
        print(f"  注意：{sid} 不在校准文件里，使用全部文件的中心")

    rng = np.random.default_rng(args.seed + fi)
    nq_ = min(args.n_queries, L - args.min_pos)
    if args.sampling == "stratified":
        seg = np.linspace(args.min_pos, L, nq_ + 1).astype(int)
        qpos = np.unique([rng.integers(seg[i], max(seg[i + 1], seg[i] + 1)) for i in range(nq_)])
    else:
        qpos = np.sort(rng.choice(np.arange(args.min_pos, L), size=nq_, replace=False))
    qpos_t = torch.tensor(qpos)
    nq = len(qpos)
    BUF["kI"] = [torch.empty(L, DI, dtype=torch.bfloat16) for _ in range(NL)]
    BUF["qI"] = [torch.empty(nq, HI, DI, dtype=torch.bfloat16) for _ in range(NL)]
    BUF["wI"] = [torch.empty(nq, HI, dtype=torch.bfloat16) for _ in range(NL)]
    BUF["qM"] = [torch.empty(nq, HM, DM, dtype=torch.bfloat16) for _ in range(NL)]
    BUF["kM"] = [torch.empty(nq, G, DM, dtype=torch.bfloat16) for _ in range(NL)]

    x = torch.tensor(ids, device=dev0)[None]
    past = None
    with torch.inference_mode():
        for s in range(0, L, args.chunk):
            e = min(L, s + args.chunk)
            m = (qpos_t >= s) & (qpos_t < e)
            ST.update(s=s, loc=qpos_t[m] - s, dst=m.nonzero().squeeze(1))
            pos = torch.arange(s, e, device=dev0)
            out = model(input_ids=x[:, s:e], attention_mask=torch.ones(1, e, dtype=torch.long, device=dev0),
                        position_ids=pos.view(1, 1, -1).expand(3, 1, -1), cache_position=pos,
                        past_key_values=past, use_cache=True)
            past = out.past_key_values
            del out
    t1 = time.time()
    k_rope_err.clear()
    pq = np.zeros((NL, len(ALLV), len(METRICS), nq), np.float32)
    for l in range(NL):
        eval_layer(l, get_keys(past, l), qpos, L, cq, ck, sid, pq)
    PQ[sid] = (qpos, L, pq)
    n_query_total += nq
    print(f"[{fi + 1}/{len(items)}] {sid}: L={L}  prefill {t1 - t0:.0f}s, 评测 {time.time() - t1:.0f}s  "
          f"主干 RoPE 自检误差 max={max(k_rope_err):.4f}（应在 1e-2 量级，bf16）")
    del past
    BUF.clear()
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- 汇总
with open(os.path.join(args.out, "eval_attn.csv"), "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["file", "layer", "variant", "topk_overlap", "indexer_mass", "attn_mass", "attn_mass_min_group"])
    wr.writerows(rows)
np.savez(os.path.join(args.out, "delta_hist.npz"), bin_labels=np.array(BIN_LABELS), n_queries=n_query_total, **hist)
np.savez(os.path.join(args.out, "per_query.npz"), variants=np.array(ALLV), metrics=np.array(METRICS),
         files=np.array(list(PQ)), topk=K,
         **{f"{sid}__qpos": v[0] for sid, v in PQ.items()},
         **{f"{sid}__len": np.array(v[1]) for sid, v in PQ.items()},
         **{f"{sid}__vals": v[2] for sid, v in PQ.items()})

allv = ["oracle", "full"] + VARS   # 打印顺序


def agg(v, metric):
    col = {"overlap": 3, "idx": 4, "attn": 5, "attn_min": 6}[metric]
    return np.array([np.mean([r[col] for r in rows if r[1] == l and r[2] == v]) for l in range(NL)])


R = {(v, m): agg(v, m) for v in allv for m in ("overlap", "idx", "attn", "attn_min")}

print(f"\n[1] 主干 dense attention 质量召回率：被选中的 {K} 个 key 覆盖的 attention 质量（GQA 组平均）")
print(f"{'layer':>5} {'oracle':>7} {'full':>7} {'drop':>7} {'table':>7} {'kmean':>7} | {'table-full':>10} {'kmean-full':>10} | 最差组: {'full':>6} {'table':>6} {'kmean':>6}")
for l in range(NL):
    print(f"{l:5d} " + " ".join(f"{R[(v, 'attn')][l]:7.4f}" for v in allv) +
          f" | {R[('table_loo', 'attn')][l] - R[('full', 'attn')][l]:+10.4f} {R[('kmean_loo', 'attn')][l] - R[('full', 'attn')][l]:+10.4f} |"
          f"      {R[('full', 'attn_min')][l]:6.4f} {R[('table_loo', 'attn_min')][l]:6.4f} {R[('kmean_loo', 'attn_min')][l]:6.4f}")
print("平均  " + " ".join(f"{R[(v, 'attn')].mean():7.4f}" for v in allv))

print("\n[2] indexer 打分质量召回率（softmax(I) 中被选中 key 的质量，相对原始 top-k）")
print(f"{'layer':>5} {'full(绝对)':>10} {'drop':>7} {'table':>7} {'kmean':>7}")
for l in range(NL):
    f_ = R[("full", "idx")][l]
    print(f"{l:5d} {f_:10.4f} " + " ".join(f"{R[(v, 'idx')][l] / f_:7.4f}" for v in VARS))
print("平均  " + f"{R[('full', 'idx')].mean():10.4f} " + " ".join(f"{(R[(v, 'idx')] / R[('full', 'idx')]).mean():7.4f}" for v in VARS))

print("\n[3] 丢失 key 的 Δ 分布（所有层合计）")
ref_c = hist["ref_cnt"].sum(0)
for v in ["table_loo", "kmean_loo"]:
    lc, lm, ac = hist[f"{v}_lost_cnt"].sum(0), hist[f"{v}_lost_mass"].sum(0), hist[f"{v}_added_cnt"].sum(0)
    print(f"  --- {v} ---  （attention 质量 = 每层每个 query 平均丢掉的 dense attention 质量）")
    print(f"  {'Δ':>14} {'原始top-k占比':>12} {'该桶丢失率':>10} {'占全部丢失':>10} {'新增占比':>9} {'丢失的attn质量':>14}")
    for b_ in range(NB):
        if ref_c[b_] == 0 and ac[b_] == 0:
            continue
        print(f"  {BIN_LABELS[b_]:>14} {ref_c[b_] / ref_c.sum():12.3%} {lc[b_] / max(ref_c[b_], 1):10.3%} "
              f"{lc[b_] / max(lc.sum(), 1):10.3%} {ac[b_] / max(ac.sum(), 1):9.3%} {lm[b_] / (n_query_total * NL):14.5f}")
    print(f"  丢失的 key 中，attention sink（位置 0~3）的个数: {int(hist[f'{v}_sink_lost'].sum())}")

pick = [("table_loo" if R[("full", "attn")][l] - R[("table_loo", "attn")][l] <= args.hybrid_tol else "kmean_loo") for l in range(NL)]
hyb = np.array([R[(pick[l], "attn")][l] for l in range(NL)])
print(f"\n[4] 混合方案（table 的 attention 召回损失 <= {args.hybrid_tol} 的层用 table，其余用 kmean）")
print(f"  用 table 的层 ({pick.count('table_loo')}): {[l for l in range(NL) if pick[l] == 'table_loo']}")
print(f"  用 kmean 的层 ({pick.count('kmean_loo')}): {[l for l in range(NL) if pick[l] == 'kmean_loo']}")
print(f"  平均 attention 召回: full={R[('full', 'attn')].mean():.4f}  混合={hyb.mean():.4f}")

# ---------------------------------------------------------------- 图
fig, ax = plt.subplots(figsize=(13, 5))
for v in allv:
    ax.plot(range(NL), R[(v, "attn")], marker="o", ms=3, label=v)
ax.set_xlabel("layer"); ax.set_ylabel(f"dense attention mass covered by top-{K}")
ax.set_title(f"Main-attention mass recall  |  bands {B} replaced  |  {len(items)} files x {args.n_queries} queries")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(args.out, "attn_recall.png"), dpi=120); plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(18, 11))
for c, v in enumerate(["table_loo", "kmean_loo"]):
    rate = hist[f"{v}_lost_cnt"] / np.maximum(hist["ref_cnt"], 1)
    im = axes[0, c].imshow(rate, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=max(rate.max(), 1e-6))
    axes[0, c].set_title(f"{v}: fraction of original top-k keys lost, by distance")
    plt.colorbar(im, ax=axes[0, c])
    mass = hist[f"{v}_lost_mass"] / max(n_query_total, 1)
    im = axes[1, c].imshow(mass, aspect="auto", origin="lower", cmap="viridis")
    axes[1, c].set_title(f"{v}: dense attention mass carried by lost keys (per query)")
    plt.colorbar(im, ax=axes[1, c])
    for r_ in range(2):
        axes[r_, c].set_xticks(range(NB)); axes[r_, c].set_xticklabels(BIN_LABELS, rotation=60, fontsize=7)
        axes[r_, c].set_xlabel("distance Δ bin"); axes[r_, c].set_ylabel("layer")
fig.tight_layout(); fig.savefig(os.path.join(args.out, "lost_delta.png"), dpi=110); plt.close(fig)
print(f"\n结果已保存到 {args.out}/: eval_attn.csv, delta_hist.npz, attn_recall.png, lost_delta.png")