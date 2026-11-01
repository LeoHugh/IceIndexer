#!/usr/bin/env python3
"""顺序前向 AA-LCR 的若干题，每题只 prefill 前 --max-tokens 个 token。

数据格式（官方 HuggingFace 仓库布局）：
  <dataset_dir>/
    AA-LCR_Dataset.csv                      # 列: document_category, document_set_id, question_id,
                                            #     question, answer, data_source_filenames(;分隔), input_tokens
    extracted_text/AA-LCR_extracted-text.zip  # 内部路径 lcr/{category}/{set_id}/{filename}

prompt 严格按 README 的官方模板拼（文档顺序按 data_source_filenames 的顺序）。

配合 indexer_qk_hook 采集 pre-RoPE q/k 时：
  1. 严格串行（并发 = 1），每个请求对应一组 dump；
  2. 用原生 /generate + input_ids 发送，token 数和 position 完全可控（0..N-1）；
  3. 每题结束写一条 manifest，可与 dump 里 positions[0]==0 的分组核对。

用法：
  python run_aa_lcr.py --data /mnt/km-nfs/jnxy/aa-lcr --inspect
  python run_aa_lcr.py --data /mnt/km-nfs/jnxy/aa-lcr --num 10 --dry-run
  python run_aa_lcr.py --data /mnt/km-nfs/jnxy/aa-lcr --num 10

服务端需要：
  --disable-cuda-graph --disable-radix-cache --skip-server-warmup --max-running-requests 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zipfile

import requests

PROMPT_TEMPLATE = """BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""


# --------------------------------------------------------------------------- 数据加载
class DocStore:
    """从 zip 或已解压目录里按 (category, set_id, filename) 取文档。"""

    def __init__(self, dataset_dir, docs_path=None):
        self._zip = None
        self._names = set()
        self._root = None

        candidates = (
            [docs_path]
            if docs_path
            else [
                os.path.join(dataset_dir, "extracted_text", "AA-LCR_extracted-text.zip"),
                os.path.join(dataset_dir, "extracted_text"),
                os.path.join(dataset_dir, "lcr"),
            ]
        )
        for cand in candidates:
            if cand and os.path.isdir(cand):
                self._root = cand
                break
            if cand and cand.endswith(".zip") and os.path.exists(cand):
                self._zip = zipfile.ZipFile(cand)
                self._names = set(self._zip.namelist())
                break
        if self._zip is None and self._root is None:
            raise SystemExit(f"找不到文档包，请用 --docs 指定 zip 或已解压目录（试过: {candidates}）")
        self._cache = {}

    def _read(self, category, set_id, filename):
        key = (category, set_id, filename)
        if key in self._cache:
            return self._cache[key]

        if self._zip is not None:
            for path in (
                f"lcr/{category}/{set_id}/{filename}",
                f"{category}/{set_id}/{filename}",
            ):
                if path in self._names:
                    text = self._zip.read(path).decode("utf-8", "replace")
                    self._cache[key] = text
                    return text
            suffix = f"/{category}/{set_id}/{filename}"
            for name in self._names:
                if name.endswith(suffix):
                    text = self._zip.read(name).decode("utf-8", "replace")
                    self._cache[key] = text
                    return text
        else:
            for path in (
                os.path.join(self._root, "lcr", category, set_id, filename),
                os.path.join(self._root, category, set_id, filename),
            ):
                if os.path.exists(path):
                    with open(path, encoding="utf-8", errors="replace") as f:
                        text = f.read()
                    self._cache[key] = text
                    return text

        raise KeyError(f"{category}/{set_id}/{filename}")

    def read_set(self, row):
        """按 data_source_filenames 的顺序拼接整套文档。"""
        category = row["document_category"].strip()
        set_id = row["document_set_id"].strip()
        filenames = [
            name.strip()
            for name in row.get("data_source_filenames", "").split(";")
            if name.strip()
        ]
        docs, missing = [], []
        for name in filenames:
            try:
                docs.append(self._read(category, set_id, name))
            except KeyError:
                missing.append(name)
        return docs, missing


def load_questions(dataset_dir):
    csv_path = os.path.join(dataset_dir, "AA-LCR_Dataset.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"找不到 {csv_path}")
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows):
        row.setdefault("question_id", str(i + 1))
    print(f"[data] {csv_path}: {len(rows)} 条问题", flush=True)
    return rows


def build_prompt(docs, question):
    documents_text = "\n\n".join(
        f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}"
        for i, doc in enumerate(docs)
    )
    return PROMPT_TEMPLATE.format(documents_text=documents_text, question=question)


# --------------------------------------------------------------------------- 分词
_TOKENIZER = None


def tokenize(base_url, text, tokenizer_path=None):
    """优先用服务端 /tokenize，保证和 server 完全一致；失败再退回本地 tokenizer。"""
    try:
        resp = requests.post(
            base_url.rstrip("/") + "/tokenize", json={"text": text}, timeout=600
        )
        if resp.ok:
            data = resp.json()
            for key in ("tokens", "input_ids", "token_ids"):
                if isinstance(data.get(key), list):
                    return data[key]
    except Exception:
        pass

    global _TOKENIZER
    if _TOKENIZER is None:
        if not tokenizer_path:
            raise SystemExit("服务端 /tokenize 不可用，且未提供 --tokenizer-path（模型目录）")
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return _TOKENIZER.encode(text, add_special_tokens=True)


# --------------------------------------------------------------------------- 推理
def health(base_url):
    for path in ("/health", "/health_generate"):
        try:
            if requests.get(base_url.rstrip("/") + path, timeout=10).ok:
                return True
        except Exception:
            continue
    return False


def prefill(base_url, input_ids):
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": 1,  # 只要 prefill；decode 那一步会被 INDEXER_QK_ONLY_PREFILL 过滤
            "temperature": 0.0,
            "ignore_eos": True,
        },
    }
    t0 = time.time()
    resp = requests.post(base_url.rstrip("/") + "/generate", json=payload, timeout=None)
    resp.raise_for_status()
    return time.time() - t0, resp.json()


def count_dumps(out_dir):
    if not out_dir or not os.path.isdir(out_dir):
        return 0
    total = 0
    for _root, _dirs, files in os.walk(out_dir):
        total += sum(1 for f in files if f.startswith("l") and f.endswith(".pt"))
    return total


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="AA-LCR 数据集目录（含 CSV 和 extracted_text/）")
    parser.add_argument("--docs", default=None, help="文档 zip 或已解压目录，默认自动探测")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--start", type=int, default=0, help="从第几条开始（0-based）")
    parser.add_argument("--qids", default=None, help="按 question_id 选，如 3,7,12（优先于 start/num）")
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--gap", type=float, default=10.0, help="两题之间的间隔秒数")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    rows = load_questions(args.data)
    store = DocStore(args.data, args.docs)

    if args.inspect:
        print(f"[inspect] 列: {sorted(rows[0].keys())}", flush=True)
        for row in rows[:3]:
            docs, missing = store.read_set(row)
            print(
                f"  qid={row['question_id']} set={row['document_category']}/"
                f"{row['document_set_id']} docs={len(docs)} missing={missing} "
                f"official_input_tokens={row.get('input_tokens')}",
                flush=True,
            )
            print(f"    question: {row['question'][:150]}", flush=True)
        return

    if args.qids:
        wanted = {q.strip() for q in args.qids.split(",")}
        batch = [r for r in rows if str(r["question_id"]).strip() in wanted]
    else:
        batch = rows[args.start : args.start + args.num]
    if len(batch) < args.num and not args.qids:
        print(f"[warn] 只取到 {len(batch)} 条（要求 {args.num}）", flush=True)

    if not args.dry_run and not health(args.base_url):
        raise SystemExit(f"服务端 {args.base_url} 不可用，先确认 server 已启动")

    out_dir = os.environ.get("INDEXER_QK_OUT_DIR")
    manifest_path = args.manifest or (
        os.path.join(out_dir, "runs.jsonl") if out_dir else "runs.jsonl"
    )

    manifest = []
    for i, row in enumerate(batch):
        docs, missing = store.read_set(row)
        if missing:
            print(f"[warn] qid={row['question_id']} 缺文档: {missing}", flush=True)
        prompt = build_prompt(docs, row["question"])
        ids_full = tokenize(args.base_url, prompt, args.tokenizer_path)
        ids = ids_full[: args.max_tokens]

        record = {
            "q_idx": args.start + i,
            "question_id": row["question_id"],
            "document_set": f"{row['document_category']}/{row['document_set_id']}",
            "prompt_chars": len(prompt),
            "full_prompt_tokens": len(ids_full),
            "sent_tokens": len(ids),
            "official_input_tokens": row.get("input_tokens"),
            "missing_docs": missing,
        }

        if args.dry_run:
            print(
                f"[dry-run] qid={row['question_id']} docs={len(docs)} "
                f"chars={len(prompt)} tokens={len(ids)} / full={len(ids_full)} "
                f"(截断到 {args.max_tokens}, 官方标注 {row.get('input_tokens')} cl100k)",
                flush=True,
            )
            manifest.append(record)
            continue

        before = count_dumps(out_dir)
        elapsed, resp = prefill(args.base_url, ids)
        time.sleep(args.gap)  # 等 dump 落盘
        after = count_dumps(out_dir)

        record.update(
            elapsed_s=round(elapsed, 2),
            dumps_added=after - before,
            reply_preview=str(resp.get("text", ""))[:60],
        )
        manifest.append(record)
        print(
            f"[{i + 1}/{len(batch)}] qid={row['question_id']} tokens={len(ids)} "
            f"prefill={elapsed:.1f}s dump+{after - before} (total {after})",
            flush=True,
        )

    with open(manifest_path, "a", encoding="utf-8") as f:
        for record in manifest:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[done] manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
