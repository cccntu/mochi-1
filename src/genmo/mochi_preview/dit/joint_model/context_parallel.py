import torch
import torch.distributed as dist
from einops import rearrange

_CONTEXT_PARALLEL_GROUP = None
_CONTEXT_PARALLEL_RANK = None
_CONTEXT_PARALLEL_GROUP_SIZE = None
_CONTEXT_PARALLEL_GROUP_RANKS = None


def get_cp_rank_size():
    if _CONTEXT_PARALLEL_GROUP:
        return _CONTEXT_PARALLEL_RANK, _CONTEXT_PARALLEL_GROUP_SIZE
    else:
        return 0, 1


def local_shard(x: torch.Tensor, dim: int = 2) -> torch.Tensor:
    if not _CONTEXT_PARALLEL_GROUP:
        return x

    cp_rank, cp_size = get_cp_rank_size()
    return x.tensor_split(cp_size, dim=dim)[cp_rank]


def set_cp_group(cp_group, ranks, global_rank):
    global _CONTEXT_PARALLEL_GROUP, _CONTEXT_PARALLEL_RANK, _CONTEXT_PARALLEL_GROUP_SIZE, _CONTEXT_PARALLEL_GROUP_RANKS
    if _CONTEXT_PARALLEL_GROUP is not None:
        raise RuntimeError("CP group already initialized.")
    _CONTEXT_PARALLEL_GROUP = cp_group
    _CONTEXT_PARALLEL_RANK = dist.get_rank(cp_group)
    _CONTEXT_PARALLEL_GROUP_SIZE = dist.get_world_size(cp_group)
    _CONTEXT_PARALLEL_GROUP_RANKS = ranks

    assert _CONTEXT_PARALLEL_RANK == ranks.index(
        global_rank
    ), f"Rank mismatch: {global_rank} in {ranks} does not have position {_CONTEXT_PARALLEL_RANK} "
    assert _CONTEXT_PARALLEL_GROUP_SIZE == len(
        ranks
    ), f"Group size mismatch: {_CONTEXT_PARALLEL_GROUP_SIZE} != len({ranks})"


def get_cp_group():
    if _CONTEXT_PARALLEL_GROUP is None:
        raise RuntimeError("CP group not initialized")
    return _CONTEXT_PARALLEL_GROUP


def is_cp_active():
    return _CONTEXT_PARALLEL_GROUP is not None


class AllGatherIntoTensorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, reduce_dtype, group: dist.ProcessGroup):
        ctx.reduce_dtype = reduce_dtype
        ctx.group = group
        ctx.batch_size = x.size(0)
        group_size = dist.get_world_size(group)

        x = x.contiguous()
        output = torch.empty(group_size * x.size(0), *x.shape[1:], dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(output, x, group=group)
        return output


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    if not _CONTEXT_PARALLEL_GROUP:
        return tensor

    return AllGatherIntoTensorFunction.apply(tensor, torch.float32, _CONTEXT_PARALLEL_GROUP)

def all_gather_async(tensor: torch.Tensor) -> torch.Tensor:
    if not _CONTEXT_PARALLEL_GROUP:
        return tensor

    tensor = tensor.contiguous()
    group_size = dist.get_world_size(_CONTEXT_PARALLEL_GROUP)
    output = torch.empty(group_size * tensor.size(0), *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)

    # Use dist directly for async all_gather
    work = dist.all_gather_into_tensor(output, tensor, group=_CONTEXT_PARALLEL_GROUP, async_op=True)
    return output, work


@torch.compiler.disable()
def _all_to_all_single(output, input, group):
    # Disable compilation since torch compile changes contiguity.
    assert input.is_contiguous(), "Input tensor must be contiguous."
    assert output.is_contiguous(), "Output tensor must be contiguous."
    return dist.all_to_all_single(output, input, group=group)


class CollectTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv: torch.Tensor, group: dist.ProcessGroup, num_heads: int):
        """Redistribute heads and receive tokens.

        Args:
            qkv: query, key or value. Shape: [B, M, 3 * num_heads * head_dim]

        Returns:
            qkv: shape: [3, B, N, local_heads, head_dim]

        where M is the number of local tokens,
        N = cp_size * M is the number of global tokens,
        local_heads = num_heads // cp_size is the number of local heads.
        """
        ctx.group = group
        ctx.num_heads = num_heads
        cp_size = dist.get_world_size(group)
        assert num_heads % cp_size == 0
        ctx.local_heads = num_heads // cp_size

        qkv = rearrange(
            qkv,
            "B M (qkv G h d) -> G M h B (qkv d)",
            qkv=3,
            G=cp_size,
            h=ctx.local_heads,
        ).contiguous()

        output_chunks = torch.empty_like(qkv)
        _all_to_all_single(output_chunks, qkv, group=group)

        return rearrange(output_chunks, "G M h B (qkv d) -> qkv B (G M) h d", qkv=3)


def all_to_all_collect_tokens(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    if not _CONTEXT_PARALLEL_GROUP:
        # Move QKV dimension to the front.
        #   B M (3 H d) -> 3 B M H d
        B, M, _ = x.size()
        x = x.view(B, M, 3, num_heads, -1)
        return x.permute(2, 0, 1, 3, 4)

    return CollectTokens.apply(x, _CONTEXT_PARALLEL_GROUP, num_heads)
@torch.compiler.disable()
def all_to_all_collect_tokens_async(qkv: torch.Tensor, num_heads: int, comm_n_heads_aat: int):
    group = _CONTEXT_PARALLEL_GROUP
    """Split all_to_all into parts, return buffers and futures."""
    B = qkv.size(0)
    G = cp_size = dist.get_world_size(group)
    h = local_heads = num_heads // cp_size
    #parts = 3  # Split 6 local heads into 3 parts of 2 heads each

    # qkv: B M (qkv H d)
    # M: total seq len
    # H: total heads
    # qkv: B M (qkv G h d)
    # G: world size
    # h: local heads
    # "B M (qkv G h d) -> G M h B (qkv d)",

    rearranged = rearrange(
        qkv,
        "B M (qkv G h d) -> G M h B (qkv d)",
        qkv=3,
        G=G,
        h=h,
    )#.contiguous()

    futures = []
    for i in range(0, h, comm_n_heads_aat):
        # dim=2
        qkv_part = rearranged.narrow(dim=2, start=i, length=comm_n_heads_aat).contiguous()
        #[:,:, i:i+1, :, :].contiguous()

        output_buffer = torch.empty_like(qkv_part)
        future = dist.all_to_all_single(
            output=output_buffer,
            input=qkv_part,
            group=group,
            async_op=True
        )
        futures.append((output_buffer, future))

    return futures
def all_to_all_collect_tokens_async_post_process(x):
    return rearrange(x, "G M h B (qkv d) -> qkv B (G M) h d", qkv=3)

# --------------------------------------------------------


class CollectHeads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup):
        """Redistribute tokens and receive heads.

        Args:
            x: Output of attention. Shape: [B, N, local_heads, head_dim]

        Returns:
            Shape: [B, M, num_heads * head_dim]
        """
        ctx.group = group
        ctx.local_heads = x.size(2)
        ctx.head_dim = x.size(3)
        group_size = dist.get_world_size(group)
        x = rearrange(x, "B (G M) h D -> G h M B D", G=group_size).contiguous()
        output = torch.empty_like(x)
        _all_to_all_single(output, x, group=group)
        del x
        return rearrange(output, "G h M B D -> B M (G h D)")


def all_to_all_collect_heads(x: torch.Tensor) -> torch.Tensor:
    if not _CONTEXT_PARALLEL_GROUP:
        # Merge heads.
        return x.view(x.size(0), x.size(1), x.size(2) * x.size(3))

    return CollectHeads.apply(x, _CONTEXT_PARALLEL_GROUP)


def all_to_all_collect_heads_async(x):
    group =_CONTEXT_PARALLEL_GROUP
    group_size = dist.get_world_size(group)

    local_heads = x.size(2)
    head_dim = x.size(3)
    x = rearrange(x, "B (G M) h D -> G h M B D", G=group_size).contiguous()
    output = torch.empty_like(x)
    future = dist.all_to_all_single(
            output=output,
            input=x,
            group=group,
            async_op=True
        )
    return output, future

def all_to_all_collect_heads_async_post_process(x):
    return rearrange(x, "G h M B D -> B M (G h D)")



