# GDR delivery-fix backport onto RCCL 2.22.3 (ROCm 6.4.4)

Diagnostic/candidate build. **Not a production release** — validated as described
below on one cluster; adopt only with explicit sign-off.

## What this branch is

Base: tag `rocm-6.4.4` (RCCL **2.22.3**, commit `2f7ac66c`, tip of
`release/rocm-rel-6.4`) — the RCCL shipped with ROCm 6.4.4, our production
baseline. On top of it, two upstream commits cherry-picked from RCCL `develop`,
unmodified except for three mechanical conflict resolutions (documented inline
in `net_ib.cc`):

| commit (upstream) | title | role |
|---|---|---|
| `60c1264d` | Improve RDMA flushing by write dummy payload with RO=0 (#1570) | the actual delivery-visibility fix |
| `5f6805b4` | RCCL Multinode DMA Buffer crash fix (#1682) | registers the new flush buffer via dmabuf; without it the fix crashes at `ibv_reg_mr` on dmabuf-only stacks (exactly what we observed) |

## Symptom

On multi-node MI300X (op2 cluster, ROCm 6.4.4 / RCCL 2.22.3) with GPU-direct
RDMA enabled (the NIC writes received data straight into VRAM), inter-node RCCL
collectives can deliver **silently corrupt payloads** when the receiving GPU is
concurrently running HBM-bandwidth-saturating compute. The receive completes,
no error is raised, but a contiguous block of the destination buffer still
holds stale data.

Application-level, this surfaced in DFT-FE (ELPA GPU eigensolvers,
Cu-nanoparticle benchmark, 2 nodes x 8 GPUs) as intermittent Cholesky failures
/ NaNs and — worse — occasional silently wrong converged energies. It was then
reproduced without the application by a "compute mule" harness: bit-exact
collective validation on one stream while a persistent HBM-saturating kernel
runs on another. On stock 2.22.3 this corrupts reliably (~8704 stale doubles
per event, contiguous block, `ncclBroadcast` receive side). The only workaround
was `NCCL_NET_GDR_LEVEL=LOC` (staging receives through host memory), which
costs ~84% of allreduce busbw.

## Root cause

A NIC-to-VRAM delivery/visibility fault on the GDR receive path. When the NIC
DMA-writes into GPU memory, RCCL must make sure those writes are actually
visible to the GPU before the proxy thread tells the GPU the data is ready.
Stock 2.22.3 does this "flush" with a single loopback `IBV_WR_RDMA_READ` of the
received data buffer: the theory is that a PCIe read cannot complete until
prior writes have landed. In practice, with relaxed-ordering (RO) PCIe traffic
on this platform, that guarantee does not hold — the flush read can be
satisfied while some of the NIC's data writes are still in flight toward HBM.
The GPU then consumes the buffer a moment too early and sees stale memory.
Heavy concurrent HBM traffic widens the write-landing window, which is why the
fault only shows up under compute load.

## What the backported commits do

### 1. RO=0 dummy-payload flush (`864cfae`, upstream #1570)

All changes are in `src/transport/net_ib.cc`:

- Each receive comm's flush machinery (`struct ncclIbGpuFlush`) gains a
  dedicated one-`int` GPU buffer (`gpuFlushGpuMem`), allocated **uncached**
  (`hipDeviceMallocUncached`, or fine-grained on older HIP) and registered as
  its own memory region (`gpuMr`). The flush QP is created with
  `IBV_ACCESS_REMOTE_WRITE` added so it can be the target of writes.
- `ncclIbIflush()` now posts **two** work requests on the loopback QP instead
  of one: first an `IBV_WR_RDMA_WRITE` of a dummy payload into the dedicated
  GPU flush buffer, then an `IBV_WR_RDMA_READ` of that same buffer back to
  host memory (this read carries the signaled completion the proxy waits on).
- Because the flush buffer's registration does not use relaxed ordering
  (RO=0), the dummy write is a strictly-ordered PCIe write: it cannot pass the
  NIC's earlier data writes, so by the time it lands in VRAM every preceding
  data write has landed too. The follow-up read of the very location just
  written can only complete after that write — so its completion proves all
  the collective's data is visible in VRAM before the GPU is signaled. The old
  flush read targeted the (relaxed-ordered) data MR itself and enforced no
  such ordering.
- Escape hatch: `RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0` restores the
  old single-read flush (the new behavior is default-on).

### 2. dmabuf registration of the flush buffer (`552acea`, upstream #1682)

Commit 1 registers its GPU flush buffer with plain `ibv_reg_mr()`, which
requires the peermem kernel path. On stacks where GDR runs over **dmabuf only**
(no peermem — our case), that registration fails and receiver setup crashes.
This commit:

- Detects dmabuf support at accept time (`useDmaBuf`) and folds it into the
  `flushEnabled` decision.
- When dmabuf is in use, page-aligns the flush buffer's address/size (new
  helpers `get_sc_page_size()` / `get_aligned_ptr_and_size()` in
  `src/misc/utils.cc`, declared in `src/include/utils.h`), exports the buffer
  as a dmabuf fd via `hsa_amd_portable_export_dmabuf`, and registers it with
  `ibv_reg_dmabuf_mr()` (iova = the buffer's device address); otherwise it
  falls back to `ibv_reg_mr()` as before.
- Tracks the fd in `ncclIbGpuFlush::dmabuf_fd` and closes it in
  `ncclIbCloseRecv()` alongside buffer free and MR deregistration.

## Validation (op2 cluster: MI300X 2 nodes x 8 GPUs, rail-aligned 8x400G IB, ROCm 6.4.4, GDR on)

Reference ground-state free energy: -5.642062658722e+04 Ha; acceptance
tolerance ~1e-7 Ha (silent wrong-energy convergence is a known failure mode of
the fault, so exit codes are never trusted alone).

| rung | test | stock 2.22.3 | this branch |
|---|---|---|---|
| a | mule reproducer, 1000 iters | corrupts (onset iter ~350-850, 8704 stale doubles/event) | **CLEAN, 0 corrupt elements, baseline speed** |
| b | DFT-FE, ELPA CCL off | corrupts intermittently | **converged 44 SCF iters, energy = reference to 1.3e-10 Ha** |
| c | DFT-FE hardest corner, ELPA GPU + ELPA CCL + DCCL, 2 replicates | corrupts intermittently | **both converged 44 SCF iters, energies = reference to 4e-10 / 7.3e-10 Ha** |
| d | `all_reduce_perf` @1 GiB busbw | 352 GB/s | **351.76 GB/s (-0.1%)** |
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
