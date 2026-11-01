export PYTHONPATH=/home/jnxy/indexer_qk_hook:$PYTHONPATH
 export INDEXER_QK_OUT_DIR=/mnt/km-nfs/jnxy/indexer_dump 
 export SGLANG_DISABLE_DSA_INDEXER_FUSION=1 
 export INDEXER_QK_ONLY_PREFILL=1 
 export INDEXER_QK_ASYNC=0 



python -m sglang.launch_server \
  --model-path /mnt/km-nfs/jnxy/modelscope-cache/models/deepseek-ai--DeepSeek-V3.2/snapshots/master  \
  --tp-size 1 --pp-size 6 \
  --disable-cuda-graph \
  --disable-radix-cache \
  --skip-server-warmup \
  --max-running-requests 1 \
  --watchdog-timeout 1800 \
  --mem-fraction-static 0.88