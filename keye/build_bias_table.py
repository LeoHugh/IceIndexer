"""
第二步：把 B 中的频段换成离线中心，预计算成只依赖距离 Δ 的位置偏置表，并验证对 top-k 选择的影响。

原始打分（每个 head，ReLU 之前）:
    s_h(i,j) = scale · Σ_{f=0}^{F-1} Re( q_{i,h,f} · conj(k_{j,f}) · e^{iω_f (i-j)} )
近似打分:
    s_h(i,j) ≈ scale · Σ_{f∉B} Re(...)  +  bias_h[i-j]
    bias_h[Δ] = scale · Σ_{f∈B} Re( q̄_{h,f} · conj(k̄_f) · e^{iω_f Δ} )
之后的 ReLU、按 w 加权求和、因果 mask、top-k 都不变。

输出:
  --table_out (safetensors)
    layer{l}.bias      [max_len, H] float32，已乘 scale，可直接加在 scale·(q_lo·k_lo) 上
    layer{l}.q_bar_hi  [H, |B|, 2]  (实部, 虚部)，可用来在线解析计算任意 Δ
    layer{l}.k_bar_hi  [|B|, 2]
    bands, omega_hi
  --out 目录: eval.csv、eval_overlap.png

验证（每个文件都用「去掉这个文件后其余文件算出的中心」，即留一，模拟没见过的数据）:
  full       原始打分（参照）
  drop       直接删掉 B 频段，不加偏置 —— 用来看偏置表到底有没有用
  table_loo  删掉 B 频段 + 留一中心算出的偏置表 —— 你的方案
  table_own  同上，但用该文件自己的中心 —— 和 table_loo 的差就是「离线中心迁移到新数据」的损失
  kmean_loo  只把 k 的 B 频段换成离线中心，q 保持动态 —— 同样能省掉这部分 K cache，但保留 query 自适应
指标: 近似 top-k 与原始 top-k 的重合率。

用法:
  python build_bias_table.py --calib calib --files "indexer_qk/*.safetensors" --bands 0-7
  python build_bias_table.py --calib calib --files "indexer_qk/*.safetensors" --bands 0-15
  python build_bias_table.py --calib calib --files "indexer_qk/*.safetensors" --bands 16-31
  python build_bias_table.py --calib calib --files "indexer_qk/*.safetensors" --bands 24-31
  python build_bias_table.py --calib calib --bands 0-7 --max_len 262144 --no_eval      # 只生成表
"""
import argparse
import collections
import csv
import glob
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


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
p.add_argument("--calib", default="calib")
p.add_argument("--files", default="indexer_qk/*.safetensors")
p.add_argument("--bands", default=None, help="默认用第一步 --bands 的值；也可以换别的频段组合直接试，不用重跑第一步")
p.add_argument("--max_len", type=int, default=65536, help="偏置表长度（最大距离）")
p.add_argument("--table_out", default="indexer_pos_bias.safetensors")
p.add_argument("--no_eval", action="store_true")
p.add_argument("--out", default="eval_bias")
p.add_argument("--eval_layers", default="all")
p.add_argument("--n_eval_files", type=int, default=0, help="0 = 全部文件")
p.add_argument("--n_queries", type=int, default=64, help="每个文件抽多少个 query 位置")
p.add_argument("--min_pos", type=int, default=4096, help="只抽这个位置之后的 query，保证 top-k 有区分度")
p.add_argument("--topk", type=int, default=2048)
p.add_argument("--qbatch", type=int, default=16)
p.add_argument("--accept", type=float, default=0.95, help="table_loo 重合率不低于这个值的层视为可以替换")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()

C = np.load(os.path.join(args.calib, "centers.npz"))
P = np.load(os.path.join(args.calib, "per_file_stats.npz"))
q_bar, k_bar, omega = C["q_bar"], C["k_bar"], C["omega"]
D = int(C["head_dim"]); F = D // 2
NL, H, _ = q_bar.shape
B = parse_bands(args.bands, F) if args.bands else [int(x) for x in C["default_bands"]]
scale = D ** -0.5
print(f"{NL} 层, {H} 个 head, head_dim={D}, 替代频段 B={B}（{2 * len(B)}/{D} 维，"
      f"即 indexer QK 计算和 K cache 的 {2 * len(B) / D:.0%}）, scale={scale}")

# ---------------------------------------------------------------- 1. 生成偏置表（用全部文件合并的中心）
d = np.arange(args.max_len)
E = np.exp(1j * np.outer(omega[B], d))                     # [|B|, max_len]
tensors = {}
for l in range(NL):
    A = q_bar[l][:, B] * np.conj(k_bar[l][B])[None]       # [H, |B|]
    T = (A @ E).real * scale                               # [H, max_len]
    tensors[f"layer{l}.bias"] = torch.from_numpy(np.ascontiguousarray(T.T.astype(np.float32)))
    qh = q_bar[l][:, B]
    tensors[f"layer{l}.q_bar_hi"] = torch.from_numpy(
        np.ascontiguousarray(np.stack([qh.real, qh.imag], -1), dtype=np.float32))
    kh = k_bar[l][B]
    tensors[f"layer{l}.k_bar_hi"] = torch.from_numpy(
        np.ascontiguousarray(np.stack([kh.real, kh.imag], -1), dtype=np.float32))
tensors["bands"] = torch.tensor(B, dtype=torch.int32)
tensors["omega_hi"] = torch.tensor(omega[B], dtype=torch.float64)
save_file(tensors, args.table_out, metadata={
    "formula": "score_h(i,j) = scale * dot(q_lo_rot, k_lo_rot) + bias[i-j, h]; then ReLU, w-weighted sum, top-k",
    "bands": ",".join(map(str, B)),
    "pairing": "band f = dims (f, f + D/2), rotate_half",
    "scale": str(scale),
    "delta": "for mRoPE: Δ = difference of the position component that these bands use (bands 0-7 -> temporal)",
})
mb = sum(t.numel() * t.element_size() for t in tensors.values()) / 1e6
print(f"偏置表已保存: {args.table_out}  ({NL} 层 × [{args.max_len}, {H}], {mb:.0f} MB)")
if args.no_eval:
    raise SystemExit

# ---------------------------------------------------------------- 2. 验证
assert args.min_pos >= args.topk, "--min_pos 必须 >= --topk，否则 top-k 等于全选"
os.makedirs(args.out, exist_ok=True)
dev = torch.device(args.device)
K = args.topk
calib_names = [str(x) for x in P["files"]]
Sq_all, Sk_all, ns = P["Sq"], P["Sk"], P["ns"]
Sq_tot, Sk_tot, n_tot = Sq_all.sum(0), Sk_all.sum(0), ns.sum()

files = sorted(glob.glob(args.files))
if args.n_eval_files > 0:
    files = files[: args.n_eval_files]
layers = list(range(NL)) if args.eval_layers == "all" else [int(x) for x in args.eval_layers.split(",")]

om_all = torch.tensor(omega, dtype=torch.float64, device=dev)
om_B = om_all[B]
m_lo = torch.ones(D, device=dev)
m_lo[B] = 0
m_lo[[b + F for b in B]] = 0


def to_z(x):
    return torch.complex(x[..., :F].contiguous(), x[..., F:].contiguous())


def template(A_hb, L):
    """A_hb: [H,|B|] complex numpy -> 模板 [H, L] float32（未乘 scale，和下面的打分同单位）"""
    dd = torch.arange(L, device=dev, dtype=torch.float64)
    Ed = torch.polar(torch.ones(len(B), L, dtype=torch.float64, device=dev), om_B[:, None] * dd[None])
    return (torch.from_numpy(A_hb).to(dev) @ Ed).real.float()


rows = []
sanity_done = False
for fi, path in enumerate(files):
    t0 = time.time()
    name = os.path.splitext(os.path.basename(path))[0]
    if name in calib_names:
        a = calib_names.index(name)
        cq_loo = (Sq_tot - Sq_all[a]) / (n_tot - ns[a]); ck_loo = (Sk_tot - Sk_all[a]) / (n_tot - ns[a])
        cq_own = Sq_all[a] / ns[a];                      ck_own = Sk_all[a] / ns[a]
    else:  # 没参与校准的文件：全部中心本身就是「没见过它」的中心
        cq_loo, ck_loo, cq_own, ck_own = q_bar, k_bar, None, None

    with safe_open(path, "pt", device="cpu") as f:
        L = f.get_slice("input_ids").get_shape()[0]
        if L <= args.min_pos:
            print(f"[跳过] {name}: L={L} <= min_pos")
            continue
        rng = np.random.default_rng(args.seed + fi)
        qpos = np.sort(rng.choice(np.arange(args.min_pos, L), size=min(args.n_queries, L - args.min_pos), replace=False))
        t_all = torch.tensor(qpos, device=dev)
        pos = torch.arange(L, device=dev, dtype=torch.float64)
        rot = torch.polar(torch.ones(L, F, dtype=torch.float64, device=dev), pos[:, None] * om_all[None]).to(torch.complex64)
        j = torch.arange(L, device=dev)

        for l in layers:
            k = f.get_tensor(f"layer{l}.k").to(dev).float()               # [L,D]
            w = f.get_tensor(f"layer{l}.w").to(dev).float()               # [L,H]
            qsl = f.get_slice(f"layer{l}.q")                              # 只读需要的 query 行
            q = torch.cat([qsl[int(i):int(i) + 1] for i in qpos]).to(dev).float()   # [n,H,D]

            zk = to_z(k) * rot
            kr = torch.cat([zk.real, zk.imag], -1)                        # RoPE 之后的 k [L,D]
            zq = to_z(q)                                                  # RoPE 之前的 q [n,H,F]
            zqr = zq * rot[t_all][:, None, :]
            qr = torch.cat([zqr.real, zqr.imag], -1)                      # [n,H,D]

            T_loo = template(cq_loo[l][:, B] * np.conj(ck_loo[l][B])[None], L)
            T_own = template(cq_own[l][:, B] * np.conj(ck_own[l][B])[None], L) if cq_own is not None else None
            kb_loo = torch.from_numpy(ck_loo[l][B]).to(dev).to(torch.complex64)

            acc = collections.defaultdict(list)
            for s in range(0, len(qpos), args.qbatch):
                tb = t_all[s:s + args.qbatch]
                qrb, wb = qr[s:s + args.qbatch], w[tb]
                S_full = torch.einsum("bhd,ld->bhl", qrb, kr)
                S_lo = torch.einsum("bhd,ld->bhl", qrb * m_lo, kr)
                causal = j[None] > tb[:, None]
                dlt = (tb[:, None] - j[None]).clamp(min=0)                # [b,L]

                def select(S):
                    I = (wb[:, :, None] * S.clamp(min=0)).sum(1).masked_fill(causal, float("-inf"))
                    return I.topk(K, dim=-1).indices

                ref = torch.zeros_like(causal).scatter_(1, select(S_full), True)

                def overlap(S):
                    return ref.gather(1, select(S)).float().mean(1)

                if not sanity_done:
                    print(f"  [自检] 原始打分自己和自己的重合率 = {overlap(S_full).mean().item():.4f}（应为 1）")
                    sanity_done = True

                acc["drop"].append(overlap(S_lo))
                acc["table_loo"].append(overlap(S_lo + T_loo[:, dlt].permute(1, 0, 2)))
                if T_own is not None:
                    acc["table_own"].append(overlap(S_lo + T_own[:, dlt].permute(1, 0, 2)))
                a_ = zq[s:s + args.qbatch][:, :, B] * kb_loo.conj()[None, None]          # [b,H,|B|]
                Eb = torch.polar(torch.ones(len(tb), len(B), L, dtype=torch.float64, device=dev),
                                 om_B[None, :, None] * dlt[:, None, :].double()).to(torch.complex64)
                acc["kmean_loo"].append(overlap(S_lo + torch.einsum("bhf,bfl->bhl", a_, Eb).real))
                del S_full, S_lo, Eb

            for v, lst in acc.items():
                rows.append((name, l, v, torch.cat(lst).mean().item()))
    print(f"[{fi + 1}/{len(files)}] {name}: L={L}  {time.time() - t0:.1f}s")

# ---------------------------------------------------------------- 3. 汇总
with open(os.path.join(args.out, "eval.csv"), "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["file", "layer", "variant", "topk_overlap"])
    wr.writerows(rows)

variants = [v for v in ["drop", "table_loo", "table_own", "kmean_loo"] if any(r[2] == v for r in rows)]
res = {v: {l: np.mean([r[3] for r in rows if r[1] == l and r[2] == v]) for l in layers} for v in variants}
worst = {l: np.min([r[3] for r in rows if r[1] == l and r[2] == "table_loo"]) for l in layers}

print(f"\ntop-{K} 重合率（{len(files)} 个文件 × {args.n_queries} 个 query 的平均；worst = table_loo 最差的文件）")
print(f"{'layer':>5} " + " ".join(f"{v:>10}" for v in variants) + f" {'worst':>8} {'迁移损失':>8} {'表的增益':>8}")
for l in layers:
    gap = res["table_own"][l] - res["table_loo"][l] if "table_own" in res else float("nan")
    gain = res["table_loo"][l] - res["drop"][l]
    print(f"{l:5d} " + " ".join(f"{res[v][l]:10.4f}" for v in variants) + f" {worst[l]:8.4f} {gap:8.4f} {gain:8.4f}")
print("平均  " + " ".join(f"{np.mean(list(res[v].values())):10.4f}" for v in variants))
ok = [l for l in layers if res["table_loo"][l] >= args.accept]
print(f"\ntable_loo >= {args.accept} 的层（{len(ok)}/{len(layers)}）: {ok}")
print(f"其余层: {[l for l in layers if l not in ok]}")

fig, ax = plt.subplots(figsize=(13, 5))
for v in variants:
    ax.plot(layers, [res[v][l] for l in layers], marker="o", ms=3, label=v)
ax.axhline(args.accept, color="gray", ls="--", lw=0.8)
ax.set_xlabel("layer"); ax.set_ylabel(f"top-{K} overlap with original indexer")
ax.set_title(f"Replacing bands {B} with offline centers  |  {len(files)} files, leave-one-out centers")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(args.out, "eval_overlap.png"), dpi=120)
print(f"结果已保存到 {args.out}/eval.csv 和 {args.out}/eval_overlap.png")
