# Fused ulysses head parallelism
# GEMM KV GEMM Q wait kv attn GEMM O
#    a2a kv  a2a q         a2a o
# use layout of (B, S, H, D) for no transpose attention
# inputs: (B * S / w, D)
# kv weight: (2, D, D) -> kv: (2, B * S / w, D)
# reshape to (2, B, S / w, H, headdim)
# a2a -> (2, B, S, H/w, headdim) (use TMA because otherwise need to call copy for S/w times)
# q weight: (D, D) -> q: (B * S / w, D)
# reshape to (B, S / w, H, headdim)
# a2a -> (B, S, H/w, headdim)
# sync to ensure kv are ready
# attn on qkv (only need to wait for each q) this attn neeed to be both consmer and producer
# consumer all2all for o (B, S, H/w, headdim) -> (B, S/w, H, headdim)
# consumer gemm for o: (B * S / w, D) -> (B * S / w, D)
# TODO: implement the full pipeline with overlapping communication and computation