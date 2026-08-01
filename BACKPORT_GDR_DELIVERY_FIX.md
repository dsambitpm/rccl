# Backport: GDR delivery/flush fix onto RCCL 2.22.3 (ROCm 6.4.4)

Diagnostic/candidate build. **Not a production release** — validated as described
below on one cluster; adopt only with explicit sign-off.

## What this branch is

Base: tag `rocm-6.4.4` (RCCL **2.22.3**, commit `2f7ac66c`) — the RCCL shipped
with ROCm 6.4.4, our production baseline.

Two upstream commits cherry-picked from RCCL `develop`, unmodified except for
three mechanical conflict resolutions (documented inline in `net_ib.cc`):

| commit (upstream) | title | why |
|---|---|---|
| `60c1264d27debb6e52286032e002e5e7d9624919` | Improve RDMA flushing by write dummy payload with RO=0 (#1570) | the actual delivery-visibility fix |
| `5f6805b4f486df79e577903f844b28ccb6ebee39` | RCCL Multinode DMA Buffer crash fix (#1682) | registers the new flush buffer via dmabuf; without it the fix crashes at `ibv_reg_mr` on dmabuf-only stacks (exactly what we observed) |

## The fault it fixes

On MI300X nodes with GPU-direct RDMA (NIC writes straight into VRAM),
inter-node RCCL collectives can deliver **silently corrupt payloads** when the
receiving GPU is concurrently running HBM-bandwidth-heavy kernels: the receive
completes, no error is raised, but a contiguous block of the destination buffer
still holds stale data. Stock 2.22.3's GDR flush is a single `IBV_WR_RDMA_READ`
of the received data buffer; under PCIe relaxed ordering that read can be
satisfied before the NIC's data writes actually land in VRAM, so the GPU reads
the buffer too early. The fix (60c1264d) allocates a dedicated
uncached/fine-grained GPU buffer and flushes by an `IBV_WR_RDMA_WRITE` of a
dummy payload followed by an `IBV_WR_RDMA_READ` of that buffer with relaxed
ordering disabled — which cannot be reordered around the preceding data writes.
Escape hatch: `RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0` restores the old
flush.

Observed app-level symptom (DFT-FE + ELPA GPU eigensolvers, Cu-nanoparticle
benchmark, 2 nodes x 8 GPUs): intermittent Cholesky failures / NaNs and, worse,
occasional silently wrong converged energies. Reproduced without the app by a
"compute mule" harness: bit-exact collective validation on one stream while a
persistent HBM-saturating kernel runs on another
(8704 corrupt doubles per event on stock 2.22.3, contiguous stale block,
`ncclBroadcast` receive side).

## Validation (op2 cluster: MI300X 2 nodes x 8 GPUs, rail-aligned 8x400G IB, ROCm 6.4.4, GDR on)

Reference ground-state free energy: -5.642062658722e+04 Ha; acceptance
tolerance ~1e-7 Ha (silent wrong-energy convergences are a known failure mode
of the fault, so exit codes are never trusted alone).

| rung | test | stock 2.22.3 | this branch |
|---|---|---|---|
| a | mule reproducer, 1000 iters | corrupts (onset iter ~350-850, 8704 stale doubles/event) | **CLEAN, 0 corrupt elements, baseline speed** |
| b | DFT-FE, ELPA CCL off (run rt49) | corrupts intermittently | **converged 44 SCF iters, energy = reference to 1.3e-10 Ha** |
| c | DFT-FE full all-RCCL corner, ELPA GPU + ELPA CCL + DCCL, 2 replicates (rt47, rt51) | corrupts intermittently | **both converged 44 SCF iters, energies = reference to 4e-10 / 7.3e-10 Ha** |
| d | `all_reduce_perf` @1 GiB busbw | 352 GB/s | **351.8 GB/s (-0.1%)** |
| d | `sendrecv_perf` @1 GiB | 22.5 GB/s | **22.48 GB/s (-0.1%)** |

Caveat discovered during validation: one cluster node
(op2-mi300x-26c3-f82c) hangs DFT-FE's first Rayleigh-Ritz eigensolve with
*every* RCCL tested — stock 2.22.3, this branch, and 2.27.7 — with identical
`deviceEventSynchronize` backtraces. That is a node fault, not a property of
this patch; runs above were on pairs excluding it (a stock-RCCL control on the
bad pair reproduced the hang, exonerating the patch).

## Build

```
./install.sh --amdgpu_targets=gfx942 -j <N> --prefix=<prefix>
```

against ROCm 6.4.4. Swap in via `LD_LIBRARY_PATH` prepend of `<prefix>/lib`
(the app binary's RUNPATH loses to it); verify with the RCCL startup banner
(`RCCL version : 2.22.3-<branch>:<sha>`).
