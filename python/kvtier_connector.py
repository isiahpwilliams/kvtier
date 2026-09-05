"""A vLLM KV connector backed by a kvtier daemon.

Block naming comes from the Rust hasher through the PyO3 bindings, never from
Python: a Python chain rule that hashed even slightly differently would not
error, it would just never hit.

The KV layout translation is the whole trick. vLLM 0.28 allocates each layer's
paged cache with logical shape

    (num_blocks, num_kv_heads, block_size, 2 * head_size)

but NHD strides, so the bytes actually run

    block -> token -> head -> K then V -> head_dim

and one (layer, block) pair is a single contiguous range. A kvtier block is
that range for every layer, concatenated in layer order. No transpose, no
gather kernel -- just one copy per layer. `BlockOrder::VllmNhd` on the Rust
side names this order and feeds it into the namespace digest, so a connector
that ever serializes differently misses instead of returning wrong bytes.
"""

from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

import kvtier
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

# Under the vllm namespace, or vLLM's logging config gives it no handler
# and every line from this module disappears.
logger = init_logger(f"vllm.{__name__}")

_TORCH_DTYPE_TO_KVTIER = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float8_e4m3fn: "f8",
    torch.float8_e5m2: "f8",
}


@dataclass
class LoadSpec:
    """Blocks to pull from the tier into vLLM's paged buffer."""

    req_id: str
    names: list[bytes]
    block_ids: list[int]


@dataclass
class SaveSpec:
    """A request's full blocks, and where this step's newly-complete ones start.

    The whole chain travels because the worker re-checks what the tier still
    holds: a block can only be admitted under a resident parent, and the tier
    may have evicted part of the prefix since the scheduler looked.
    """

    req_id: str
    names: list[bytes]
    block_ids: list[int]
    depths: list[int]
    first: int


@dataclass
class KvtierMetadata(KVConnectorMetadata):
    loads: list[LoadSpec] = field(default_factory=list)
    saves: list[SaveSpec] = field(default_factory=list)


@dataclass
class _ReqState:
    """Scheduler-side bookkeeping, from the lookup until the request ends."""

    names: list[bytes] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    num_local_blocks: int = 0
    #: Blocks already handed to the tier, so a chunked prefill offers each once.
    offered_blocks: int = 0


class _Saver:
    """Writes blocks to the tier off the forward pass.

    The GPU read stays on the caller's thread, inside the forward context
    where the paged blocks are still valid, and only the socket write is
    deferred. `wait_for_save` therefore still means what vLLM needs it to
    mean -- nothing touches the paged buffer once it returns -- while the
    wire, which is the expensive half, leaves the critical path.

    One worker thread, deliberately. Puts have to land in chain order: a
    block is only admitted under a resident parent, so a second thread could
    orphan a chain it happened to overtake.
    """

    def __init__(self, address: str, slab_bytes: int, slabs: int):
        # Its own connection. The Rust client is a single in-order stream and
        # takes &mut self, so it cannot be shared with the load path.
        self._client = kvtier.Client(address)
        self._slabs = []
        self._free: queue.Queue[int] = queue.Queue()
        for index in range(slabs):
            slab = torch.empty(slab_bytes, dtype=torch.uint8)
            page_lock(slab)
            self._slabs.append(slab)
            self._free.put(index)

        self._jobs: queue.Queue = queue.Queue()
        self._inflight: set[bytes] = set()
        self._lock = threading.Lock()
        self.stored = 0
        self.dropped = 0
        self._thread = threading.Thread(
            target=self._run, name="kvtier-save", daemon=True
        )
        self._thread.start()

    def acquire(self):
        """A free slab, blocking if every one is still in flight."""
        index = self._free.get()
        return index, self._slabs[index]

    def submit(self, index: int, parent, names) -> None:
        with self._lock:
            self._inflight.update(name for name, _ in names)
        self._jobs.put((index, parent, names))

    def holds(self, name: bytes) -> bool:
        """Queued but not yet admitted, so worth treating as already stored."""
        with self._lock:
            return name in self._inflight

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            index, parent, names = job
            try:
                inserted, _, dropped = self._client.put_from(
                    parent, names, self._slabs[index].numpy()
                )
                self.stored += inserted
                self.dropped += dropped
            except Exception:
                # A failed save is a future cache miss, never a wrong answer.
                logger.exception("kvtier: a background save failed")
            finally:
                with self._lock:
                    self._inflight.difference_update(name for name, _ in names)
                self._free.put(index)

    def close(self) -> None:
        self._jobs.put(None)
        self._thread.join(timeout=30)


class KvtierConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )

        groups = kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            raise ValueError(
                f"kvtier handles one full-attention KV group, got {len(groups)}. "
                "Hybrid models are out of scope."
            )
        self._layer_names = list(groups[0].layer_names)

        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        self._block_size = cache_config.block_size
        self._tp_size = parallel_config.tensor_parallel_size
        self._tp_rank = get_tp_rank(role, parallel_config)

        dtype = model_config.dtype
        if dtype not in _TORCH_DTYPE_TO_KVTIER:
            raise ValueError(f"kvtier has no name for dtype {dtype}")

        address = self._kv_transfer_config.get_from_extra_config(
            "kvtier_address", os.environ.get("KVTIER_ADDR", "127.0.0.1:7431")
        )
        self._address = address
        self._client = kvtier.Client(address)
        self._hasher = self._client.hasher(tp_rank=self._tp_rank, tp_size=self._tp_size)

        expected = (
            2
            * len(self._layer_names)
            * self._block_size
            * model_config.get_num_kv_heads(parallel_config)
            * model_config.get_head_size()
            * dtype.itemsize
        )
        served = model_config.model
        if self._client.model_id != served:
            raise ValueError(
                f"kvtierd is namespaced for model {self._client.model_id!r}, this "
                f"engine serves {served!r}. Identical layouts would otherwise "
                "share names across models."
            )
        if self._client.block_bytes != expected:
            raise ValueError(
                f"kvtierd holds {self._client.block_bytes} B blocks, this engine "
                f"needs {expected} B. Start the daemon with the model's layout."
            )

        # Worker-side state, filled in by register_kv_caches.
        self._blocks_view: list[torch.Tensor] = []
        self._elems_per_block = 0
        self._slab_blocks = int(
            self._kv_transfer_config.get_from_extra_config("kvtier_slab_blocks", 256)
        )
        self._save_slabs = int(
            self._kv_transfer_config.get_from_extra_config("kvtier_save_slabs", 2)
        )
        self._async_saves = bool(
            int(self._kv_transfer_config.get_from_extra_config("kvtier_async_saves", 1))
        )
        self._slab = torch.empty(0, dtype=torch.uint8)
        self._slab_locked = False
        self._saver: "_Saver | None" = None
        self._verify = bool(int(os.environ.get("KVTIER_VERIFY", "0")))
        # Benchmark control: stay installed, serve nothing, store nothing.
        # Separates "the tier moved KV" from "a connector was present", which
        # otherwise both change with the same flag.
        self._inert = bool(int(os.environ.get("KVTIER_INERT", "0")))

        self._reqs: dict[str, _ReqState] = {}
        self._pending_loads: list[LoadSpec] = []

        logger.info(
            "kvtier connector: %s role=%s block=%d B tp=%d/%d",
            address,
            role.name,
            self._client.block_bytes,
            self._tp_rank,
            self._tp_size,
        )

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        # Pins the byte order the namespace digest names. Without this the
        # engine could pick HND and every block would be silently reordered.
        return "NHD"

    # ==============================
    # Worker side
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        views = []
        for name in self._layer_names:
            tensor = kv_caches[name]
            per_block = tensor.stride(0)
            if per_block != tensor[0].numel() or per_block != max(tensor.stride()):
                raise ValueError(
                    f"layer {name} does not store a block contiguously: "
                    f"shape {tuple(tensor.shape)} stride {tuple(tensor.stride())}"
                )
            # Collapse everything below the block index. The result is the
            # block's bytes in whatever order the backend chose, which under
            # the NHD layout we require is the order the namespace promises.
            views.append(
                tensor.as_strided((tensor.shape[0], per_block), (per_block, 1))
            )
            self._elems_per_block = per_block
        self._blocks_view = views
        self._device = views[0].device
        self._kv_dtype = views[0].dtype
        self._make_slab()
        logger.info(
            "kvtier registered %d layers, %d elements per (layer, block), "
            "%d-block slab%s, saves %s",
            len(views),
            self._elems_per_block,
            self._slab_blocks,
            " (page-locked)" if self._slab_locked else " (NOT page-locked)",
            f"async over {self._save_slabs} slabs" if self._saver else "synchronous",
        )

    def _make_slab(self) -> None:
        """Host staging, page-locked so copies are DMA.

        The load path owns one slab and reuses it immediately. The save path
        needs its own, and more than one, because a slab stays busy until its
        write reaches the tier.
        """
        nbytes = self._slab_blocks * self._client.block_bytes
        self._slab = torch.empty(nbytes, dtype=torch.uint8)
        self._slab_locked = page_lock(self._slab)
        if self._async_saves:
            self._saver = _Saver(self._address, nbytes, self._save_slabs)
        if not self._slab_locked:
            logger.warning(
                "kvtier could not page-lock its %d B slab; host copies will "
                "stage through the driver instead of going straight to DMA",
                nbytes,
            )

    def _staged(self, slab: torch.Tensor, blocks: int) -> torch.Tensor:
        """The first `blocks` blocks of `slab`, shaped (block, layer, elem)."""
        span = blocks * self._client.block_bytes
        return (
            slab[:span]
            .view(self._kv_dtype)
            .view(blocks, len(self._blocks_view), self._elems_per_block)
        )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        meta = self._get_connector_metadata()
        assert isinstance(meta, KvtierMetadata)
        for spec in meta.loads:
            self._load_blocks(spec)

    def _load_blocks(self, spec: LoadSpec) -> None:
        done = 0
        while done < len(spec.names):
            names = spec.names[done : done + self._slab_blocks]
            got = self._client.fetch_into(names, self._slab.numpy())
            if got:
                staged = self._staged(self._slab, got)
                ids = torch.tensor(
                    spec.block_ids[done : done + got], device=self._device
                )
                # One contiguous host-to-device copy out of page-locked memory,
                # then scatter on the GPU. Copying each layer's slice separately
                # would hand the driver a strided host view, which it stages
                # through a pageable temporary -- slower, and unsafe to do
                # non-blocking because the temporary can outlive the copy.
                resident = staged.to(self._device, non_blocking=True)
                for layer, view in enumerate(self._blocks_view):
                    view.index_copy_(0, ids, resident[:, layer, :])
                torch.cuda.current_stream(self._device).synchronize()
                if self._verify:
                    self._verify_load(spec, done, got, ids)
            done += got
            if got < len(names):
                # The tier ran out of resident blocks; the rest gets recomputed.
                break

    def wait_for_layer_load(self, layer_name: str) -> None:
        return  # start_load_kv is synchronous

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        return  # a kvtier block spans every layer, so saving happens once, below

    def _verify_save(self, spec, at, stop, ids, slab):
        """Debug: the slab must equal what is actually in the paged buffer."""
        gathered = torch.stack(
            [v.index_select(0, ids) for v in self._blocks_view], dim=1
        ).cpu()
        if not torch.equal(gathered, self._staged(slab, stop - at)):
            logger.error(
                "kvtier VERIFY save MISMATCH: %s blocks %d..%d", spec.req_id, at, stop
            )
        else:
            logger.info(
                "kvtier VERIFY save ok: %s blocks %d..%d ids=%s",
                spec.req_id, at, stop, ids.tolist()[:6],
            )

    def _verify_load(self, spec, done, got, ids):
        """Debug: the paged buffer must equal what was fetched into the slab."""
        gathered = torch.stack(
            [v.index_select(0, ids) for v in self._blocks_view], dim=1
        ).cpu()
        staged = self._staged(self._slab, got)
        if not torch.equal(gathered, staged):
            wrong = (gathered != staged).flatten(1).any(dim=1).nonzero().flatten()
            logger.error(
                "kvtier VERIFY load MISMATCH: %s %d of %d blocks wrong: %s",
                spec.req_id, wrong.numel(), got, wrong.tolist()[:6],
            )
        else:
            logger.info(
                "kvtier VERIFY load ok: %s %d blocks ids=%s",
                spec.req_id, got, ids.tolist()[:6],
            )

    def wait_for_save(self):
        meta = self._get_connector_metadata()
        assert isinstance(meta, KvtierMetadata)
        for spec in meta.saves:
            self._save_blocks(spec)

    def _save_blocks(self, spec: SaveSpec) -> None:
        # The tier may have evicted part of the prefix since the scheduler
        # looked. A block is only admitted under a resident parent, so ask
        # again and restart the run wherever the chain still ends.
        held = self._client.match_prefix(spec.names)
        # Blocks already queued count as held: the worker drains in order, so
        # they will be admitted before anything parented on them.
        while self._saver is not None and held < len(spec.names):
            if not self._saver.holds(spec.names[held]):
                break
            held += 1
        if held >= len(spec.names):
            return
        parent = spec.names[held - 1] if held else None

        at = held
        while at < len(spec.names):
            stop = min(at + self._slab_blocks, len(spec.names))
            if self._saver is not None:
                index, slab = self._saver.acquire()
            else:
                index, slab = None, self._slab

            ids = torch.tensor(spec.block_ids[at:stop], device=self._device)
            staged = self._staged(slab, stop - at)
            # Gather every layer on the GPU first, so the way back is a single
            # contiguous device-to-host copy into page-locked memory.
            gathered = torch.stack(
                [view.index_select(0, ids) for view in self._blocks_view], dim=1
            )
            staged.copy_(gathered)
            torch.cuda.current_stream(self._device).synchronize()
            if self._verify:
                self._verify_save(spec, at, stop, ids, slab)

            names = [(spec.names[i], spec.depths[i]) for i in range(at, stop)]
            if self._saver is not None:
                # The paged buffer has been read; the rest is host memory.
                self._saver.submit(index, parent, names)
            else:
                _, _, dropped = self._client.put_from(parent, names, slab.numpy())
                if dropped:
                    logger.debug(
                        "kvtier dropped %d blocks for %s", dropped, spec.req_id
                    )
                    return
            parent = spec.names[stop - 1]
            at = stop

    def shutdown(self):
        if self._saver is not None:
            self._saver.close()
            self._saver = None

    # ==============================
    # Scheduler side
    # ==============================

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        prompt = request.prompt_token_ids
        if not prompt or self._inert:
            return 0, False

        names = self._hasher.chain(prompt)
        # One token must be left over to actually run a forward pass on, so the
        # last block of an exactly-block-aligned prompt is never a load target.
        checkable = (len(prompt) - 1) // self._block_size
        local_blocks = num_computed_tokens // self._block_size

        held = self._client.match_prefix(names[:checkable]) if checkable else 0
        external_blocks = max(0, held - local_blocks)

        self._reqs[request.request_id] = _ReqState(
            names=names, num_local_blocks=local_blocks
        )
        if self._verify:
            logger.info(
                "kvtier VERIFY lookup: %s prompt=%d blocks=%d local=%d held=%d ext=%d",
                request.request_id, len(prompt), len(names), local_blocks, held,
                external_blocks,
            )
        return external_blocks * self._block_size, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        state = self._reqs.get(request.request_id)
        if state is None:
            return

        state.block_ids = list(blocks.get_block_ids()[0])
        if num_external_tokens == 0:
            return

        start = state.num_local_blocks
        stop = start + num_external_tokens // self._block_size
        self._pending_loads.append(
            LoadSpec(
                req_id=request.request_id,
                names=state.names[start:stop],
                block_ids=state.block_ids[start:stop],
            )
        )
        # Loaded blocks are already in the tier; never offer them back.
        state.offered_blocks = max(state.offered_blocks, stop)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        saves = []
        for new_req in scheduler_output.scheduled_new_reqs:
            spec = self._blocks_completed_this_step(
                new_req.req_id,
                new_req.num_computed_tokens,
                scheduler_output.num_scheduled_tokens.get(new_req.req_id, 0),
                new_req.block_ids[0],
            )
            if spec is not None:
                saves.append(spec)

        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            new_blocks = cached.new_block_ids[i]
            state = self._reqs.get(req_id)
            if state is not None and new_blocks is not None:
                if req_id in cached.resumed_req_ids:
                    state.block_ids = list(new_blocks[0])
                else:
                    state.block_ids.extend(new_blocks[0])
            spec = self._blocks_completed_this_step(
                req_id,
                cached.num_computed_tokens[i],
                scheduler_output.num_scheduled_tokens.get(req_id, 0),
                None,
            )
            if spec is not None:
                saves.append(spec)

        meta = KvtierMetadata(loads=self._pending_loads, saves=saves)
        self._pending_loads = []
        return meta

    def _blocks_completed_this_step(
        self,
        req_id: str,
        num_computed_tokens: int,
        num_scheduled_tokens: int,
        block_ids: list[int] | None,
    ) -> SaveSpec | None:
        """Blocks that become full during this step and are not in the tier yet."""
        state = self._reqs.get(req_id)
        if state is None or self._inert:
            return None
        if block_ids is not None:
            state.block_ids = list(block_ids)

        end = num_computed_tokens + num_scheduled_tokens
        # Only full blocks are named, and only prompt blocks were named at all.
        complete = min(end // self._block_size, len(state.names))
        start = state.offered_blocks
        if complete <= start:
            return None

        names = state.names[start:complete]
        ids = state.block_ids[start:complete]
        if len(ids) != len(names):
            return None  # blocks not allocated yet; catch it next step
        state.offered_blocks = complete
        if self._verify:
            logger.info(
                "kvtier VERIFY save-plan: %s computed=%d sched=%d complete=%d "
                "first=%d n_block_ids=%d",
                req_id, num_computed_tokens, num_scheduled_tokens, complete, start,
                len(state.block_ids),
            )

        return SaveSpec(
            req_id=req_id,
            names=state.names[:complete],
            block_ids=state.block_ids[:complete],
            depths=[(i + 1) * self._block_size for i in range(complete)],
            first=start,
        )

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        self._reqs.pop(request.request_id, None)
        return False, None


def page_lock(tensor: torch.Tensor) -> bool:
    """cudaHostRegister the tensor's storage, so the GPU can DMA in and out.

    Pageable host memory forces the driver to bounce every transfer through
    its own pinned staging buffer. Registering removes that copy.
    """
    try:
        cudart = torch.cuda.cudart()
        # cudaHostRegisterPortable | cudaHostRegisterMapped
        status = cudart.cudaHostRegister(tensor.data_ptr(), tensor.numel(), 3)
        return int(status) == 0
    except Exception as error:  # no CUDA, or already registered
        logger.debug("kvtier could not page-lock the slab: %s", error)
        return False


def get_tp_rank(role: KVConnectorRole, parallel_config) -> int:
    """Rank 0 on the scheduler, which only ever names blocks it will not touch."""
    if role is KVConnectorRole.SCHEDULER:
        return 0
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

    return get_tensor_model_parallel_rank()
