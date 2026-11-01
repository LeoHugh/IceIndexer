"""让每个 Python 解释器（含 PP 子进程）自动装上 DSA indexer q/k 采集 hook。

用法：把本目录加入 PYTHONPATH，并设置 INDEXER_QK_OUT_DIR。
只有设置了 INDEXER_QK_OUT_DIR 才会生效，其余进程零开销。
"""

import os

if os.environ.get("INDEXER_QK_OUT_DIR"):
    try:
        import dsa_qk_dump

        dsa_qk_dump.install()
    except Exception as exc:  # 采集失败绝不能阻断 server 启动
        print(f"[dsa_qk_dump] install failed: {exc!r}", flush=True)
