"""
第一步：逐个文件计算 indexer pre-RoPE q / k 的复数中心，比较不同文件之间的差异，得到最终的离线中心。

频段 f（0..F-1, F=32）对应复数 z_f = x[f] + i·x[f+F]（rotate_half 配对），频率 ω_f = theta^(-2f/D)。
中心: q̄[l,h,f] = E_t[z^q_{t,h,f}]，k̄[l,f] = E_t[z^k_{t,f}]（k 是 MQA 共享的一个 head）。
位置偏置模板只依赖两者的乘积 A[l,h,f] = q̄ · conj(k̄)：
    T_lh(Δ) = Σ_{f∈B} Re( A[l,h,f] · e^{iω_f Δ} )
所以比较文件间差异时，除了 q̄、k̄ 各自的漂移，更直接的指标是 A 的差异（模板误差）。

输出（--out 目录下）:
  per_file_stats.npz      每个文件的累计量（复数和、模长和、模长平方和、token 数），第二步做留一验证要用
  centers.npz             用全部文件按 token 加权合并得到的最终中心 q_bar [NL,H,F]、k_bar [NL,F]（所有频段都存）
  template_loo_error.csv  每个文件、每层的留一模板误差
  calib_report.png        汇总图

用法:
  python calib_centers.py --files "indexer_qk/*.safetensors" --out calib --bands 0-7 --meta lbv2_50k.jsonl
"""
import argparse
import csv
import glob
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open


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
p.add_argument("--files", default="indexer_qk/*.safetensors")
p.add_argument("--out", default="calib")
p.add_argument("--bands", default="0-7", help="准备用离线值替代的频段，如 0-7 或 0,1,2,5")
p.add_argument("--theta", type=float, default=1e7)
p.add_argument("--meta", default=None, help="prep_longbench_v2.py 输出的 jsonl，用来按 domain 分组")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = torch.device(args.device)

files = sorted(glob.glob(args.files))
assert files, f"没有匹配到文件: {args.files}"
names = [os.path.splitext(os.path.basename(x))[0] for x in files]
with safe_open(files[0], "pt") as f:
    keys = list(f.keys())
    _, H, D = f.get_slice("layer0.q").get_shape()
NL = 1 + max(int(k.split(".")[0][5:]) for k in keys if k.startswith("layer"))
F = D // 2
B = parse_bands(args.bands, F)
omega = args.theta ** (-2.0 * np.arange(F) / D)
print(f"{len(files)} 个文件, {NL} 层, {H} 个 q head, head_dim={D}, 替代频段 B={B}, "
      f"周期 {2 * np.pi / omega[B[0]]:.1f} ~ {2 * np.pi / omega[B[-1]]:.1f} token")

domain = {}
if args.meta:
    with open(args.meta, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ex = json.loads(line)
                domain[str(ex["id"])] = ex.get("domain", "?")
dom = [domain.get(n, "?") for n in names]

# ---------------------------------------------------------------- 1. 逐文件累计
N = len(files)
Sq = np.zeros((N, NL, H, F), np.complex128); Aq = np.zeros((N, NL, H, F)); Pq = np.zeros((N, NL, H, F))
Sk = np.zeros((N, NL, F), np.complex128);    Ak = np.zeros((N, NL, F));    Pk = np.zeros((N, NL, F))
ns = np.zeros(N, np.int64)


def to_z(x):  # [..., D] float -> complex [..., F]
    return torch.complex(x[..., :F].contiguous(), x[..., F:].contiguous())


for a, path in enumerate(files):
    t0 = time.time()
    with safe_open(path, "pt", device="cpu") as f:
        ns[a] = f.get_slice("input_ids").get_shape()[0]
        for l in range(NL):
            zq = to_z(f.get_tensor(f"layer{l}.q").to(dev).float())   # [L,H,F]
            zk = to_z(f.get_tensor(f"layer{l}.k").to(dev).float())   # [L,F]
            mq, mk = zq.abs(), zk.abs()
            Sq[a, l] = zq.sum(0).cpu().numpy(); Aq[a, l] = mq.sum(0).cpu().numpy(); Pq[a, l] = (mq * mq).sum(0).cpu().numpy()
            Sk[a, l] = zk.sum(0).cpu().numpy(); Ak[a, l] = mk.sum(0).cpu().numpy(); Pk[a, l] = (mk * mk).sum(0).cpu().numpy()
    print(f"  [{a + 1:2d}/{N}] {names[a]}  L={ns[a]}  {dom[a]}  {time.time() - t0:.1f}s")

np.savez(os.path.join(args.out, "per_file_stats.npz"), Sq=Sq, Aq=Aq, Pq=Pq, Sk=Sk, Ak=Ak, Pk=Pk, ns=ns,
         files=np.array(names), domains=np.array(dom))

# ---------------------------------------------------------------- 2. 合并得到最终中心（按 token 加权）
n_tot = ns.sum()
q_bar, k_bar = Sq.sum(0) / n_tot, Sk.sum(0) / n_tot              # [NL,H,F], [NL,F]
Eq_abs, Ek_abs = Aq.sum(0) / n_tot, Ak.sum(0) / n_tot
Eq_sq, Ek_sq = Pq.sum(0) / n_tot, Pk.sum(0) / n_tot
Rq, Rk = np.abs(q_bar) / Eq_abs, np.abs(k_bar) / Ek_abs
np.savez(os.path.join(args.out, "centers.npz"), q_bar=q_bar, k_bar=k_bar, Rq=Rq, Rk=Rk,
         Eq_abs=Eq_abs, Ek_abs=Ek_abs, Eq_sq=Eq_sq, Ek_sq=Ek_sq, omega=omega, theta=args.theta,
         head_dim=D, n_tok=n_tot, files=np.array(names), default_bands=np.array(B))

# ---------------------------------------------------------------- 3. 文件间差异
own_q = Sq / ns[:, None, None, None]
own_k = Sk / ns[:, None, None]
loo_q = (Sq.sum(0)[None] - Sq) / (n_tot - ns)[:, None, None, None]     # 去掉文件 a 后其余文件的中心
loo_k = (Sk.sum(0)[None] - Sk) / (n_tot - ns)[:, None, None]


def tmpl(cq, ck):  # [...,NL,H,F], [...,NL,F] -> 模板系数 A，只取 B 中的频段 [...,NL,H,|B|]
    return cq[..., B] * np.conj(ck[..., None, :][..., B])


A_own, A_loo = tmpl(own_q, own_k), tmpl(loo_q, loo_k)
# 不同频率的余弦在长距离上近似正交，所以系数的相对 L2 误差 ≈ 模板 T(Δ) 本身的相对 RMS 误差
err = np.sqrt((np.abs(A_own - A_loo) ** 2).sum((2, 3)) / (np.abs(A_loo) ** 2).sum((2, 3)))   # [N,NL]
# 位置项占比：B 中频段的打分里，只依赖 Δ 的那部分（模板）的能量占比；越接近 1，用离线值替代损失越小
pos_share = (np.abs(q_bar[:, :, B]) ** 2 * np.abs(k_bar[:, None, B]) ** 2).sum(-1) / \
            (Eq_sq[:, :, B] * Ek_sq[:, None, B]).sum(-1)                                        # [NL,H]
drift_q = np.abs(own_q - loo_q) / Eq_abs[None]     # 中心漂移，以该频段的平均模长为单位 [N,NL,H,F]
drift_k = np.abs(own_k - loo_k) / Ek_abs[None]     # [N,NL,F]

with open(os.path.join(args.out, "template_loo_error.csv"), "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["file", "domain"] + [f"layer{l}" for l in range(NL)])
    for a in range(N):
        wr.writerow([names[a], dom[a]] + [f"{e:.5f}" for e in err[a]])

print("\n逐层汇总（B 频段）：")
print("  pos_share = 模板能量 / 该频段打分总能量（head 平均，括号内为最小 head）")
print("  tmpl_err  = 单个文件的模板 vs 其余 17 个文件的模板，相对 RMS 误差（文件中位数 / 最大值）")
print("  drift     = 中心漂移 / 平均模长（所有文件、head、B 频段中的最大值）")
print(f"{'layer':>5} {'pos_share':>16} {'Rq(B)':>6} {'Rk(B)':>6} {'tmpl_err med':>12} {'max':>7}  {'drift_q':>7} {'drift_k':>7}")
for l in range(NL):
    ps = pos_share[l]
    print(f"{l:5d} {ps.mean():8.3f} ({ps.min():.3f}) {Rq[l][:, B].mean():6.3f} {Rk[l][B].mean():6.3f} "
          f"{np.median(err[:, l]):12.4f} {err[:, l].max():7.4f}  "
          f"{drift_q[:, l][..., B].max():7.4f} {drift_k[:, l][..., B].max():7.4f}")

print("\n逐文件（模板误差在各层上的平均）：")
order = np.argsort(-err.mean(1))
for a in order:
    print(f"  {names[a]:<40s} {dom[a]:<36s} mean={err[a].mean():.4f}  max={err[a].max():.4f} (layer {err[a].argmax()})")
if args.meta:
    print("\n按 domain：")
    for d in sorted(set(dom)):
        m = np.array(dom) == d
        print(f"  {d:<40s} 文件数={m.sum():2d}  模板误差均值={err[m].mean():.4f}")

print(f"\n整体: 模板留一误差 中位数={np.median(err):.4f}  95分位={np.quantile(err, 0.95):.4f}  最大={err.max():.4f}")
print(f"      pos_share 均值={pos_share.mean():.3f}；pos_share<0.5 的 (层,head) 占比 {np.mean(pos_share < 0.5):.3f}")
print(f"最终中心已保存到 {os.path.join(args.out, 'centers.npz')}（{N} 个文件、{n_tot} 个 token 按 token 加权合并，所有频段都保存）")

# ---------------------------------------------------------------- 4. 图
fig, axes = plt.subplots(2, 2, figsize=(18, 13))
im = axes[0, 0].imshow(pos_share, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
axes[0, 0].set_title(f"position-only share of band-{B[0]}..{B[-1]} score energy  |q̄|²|k̄|² / E|q|²E|k|²")
axes[0, 0].set_xlabel("indexer head"); axes[0, 0].set_ylabel("layer"); plt.colorbar(im, ax=axes[0, 0])

ax = axes[0, 1]
doms = sorted(set(dom))
cmap = plt.get_cmap("tab10")
rng = np.random.default_rng(0)
for di, d in enumerate(doms):
    for a in np.where(np.array(dom) == d)[0]:
        ax.scatter(np.arange(NL) + rng.uniform(-0.25, 0.25, NL), err[a], s=6, color=cmap(di % 10),
                   label=d if a == np.where(np.array(dom) == d)[0][0] else None)
ax.plot(np.arange(NL), np.median(err, 0), color="k", lw=1.2, label="median over files")
ax.set_xlabel("layer"); ax.set_ylabel("relative RMS error of template")
ax.set_title("leave-one-out template error: file's own template vs. template from the other files")
ax.grid(alpha=0.3); ax.legend(fontsize=7, markerscale=2)

for ax, mat, title in [(axes[1, 0], drift_q.max(0).mean(1), "q center drift (max over files, mean over heads)"),
                       (axes[1, 1], drift_k.max(0), "k center drift (max over files)")]:
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(title + "  / E|z|"); ax.set_xlabel("band f (high freq -> low freq)"); ax.set_ylabel("layer")
    ax.axvspan(B[0] - 0.5, B[-1] + 0.5, fill=False, edgecolor="cyan", lw=1.5)
    plt.colorbar(im, ax=ax)
fig.suptitle(f"Offline center calibration  |  {N} files, {n_tot} tokens  |  replaced bands B = {B} (cyan box)")
fig.tight_layout()
fig.savefig(os.path.join(args.out, "calib_report.png"), dpi=110)
print(f"图已保存到 {os.path.join(args.out, 'calib_report.png')}")