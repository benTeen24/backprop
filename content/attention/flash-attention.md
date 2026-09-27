---
title: Flash Attention
---

Notes on Flash Attention — the IO-aware exact attention algorithm.

## Why it matters

Standard attention materializes the full $N \times N$ attention matrix, which is $O(N^2)$ in memory and dominated by HBM read/write traffic rather than compute. Flash Attention restructures the computation to avoid ever writing that matrix to HBM.

## Core idea

- Tile the $Q$, $K$, $V$ matrices into blocks that fit in on-chip SRAM
- Compute attention output incrementally per block, using online softmax (running max + running sum) so the full softmax normalization never needs the whole row materialized at once
- Recompute (rather than store) intermediate values during the backward pass, trading compute for memory bandwidth

$$
O = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

## To read next

- [ ] Flash Attention 2 — better parallelism across sequence length
- [ ] Flash Attention 3 — FP8 + async pipelining on Hopper
