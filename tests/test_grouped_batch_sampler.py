# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from lightning_fabric.utilities.data import _replace_dunder_methods, _update_dataloader
import torch
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from instant_nurec.training.data import GroupedSkipBatchSampler

GROUPS = (7, 5)  # two mixture components: indices 0-6 and 7-11


def _group(index: int) -> int:
    return 0 if index < GROUPS[0] else 1


class _Distributed:
    def __init__(self, num_replicas: int, rank: int) -> None:
        self.num_replicas, self.rank = num_replicas, rank


def _sampler(base=None, **kwargs) -> GroupedSkipBatchSampler:
    kwargs.setdefault("batch_size", 2)
    return GroupedSkipBatchSampler(base or SequentialSampler(range(sum(GROUPS))), group_sizes=GROUPS, seed=38, **kwargs)


def test_every_batch_comes_from_one_group_and_covers_each_index_once() -> None:
    batches = list(_sampler())

    assert all(len({_group(i) for i in batch}) == 1 for batch in batches)
    assert sorted(i for batch in batches for i in batch) == list(range(sum(GROUPS)))
    assert len(_sampler()) == len(batches) == 4 + 3


def test_ranks_get_equal_batch_counts_covering_all_indices() -> None:
    for world_size in (2, 3, 4):
        per_rank = [list(_sampler(_Distributed(world_size, rank))) for rank in range(world_size)]

        assert len({len(batches) for batches in per_rank}) == 1
        assert all(len(_sampler(_Distributed(world_size, r))) == len(per_rank[r]) for r in range(world_size))
        assert {i for batches in per_rank for batch in batches for i in batch} == set(range(sum(GROUPS)))
        assert all(len({_group(i) for i in batch}) == 1 for batches in per_rank for batch in batches)


def test_order_depends_only_on_seed_and_epoch() -> None:
    assert list(_sampler(epoch=0)) == list(_sampler(epoch=0))
    assert list(_sampler(epoch=0)) != list(_sampler(epoch=1))


def test_resume_skips_completed_batches_but_keeps_epoch_length() -> None:
    full = list(_sampler())
    resumed = _sampler(first_batch_idx=3)

    assert list(resumed) == full[3:]
    assert len(resumed) == len(full)


def test_drop_last_drops_each_groups_partial_batch() -> None:
    batches = list(_sampler(drop_last=True))

    assert all(len(batch) == 2 for batch in batches)
    assert len(batches) == len(_sampler(drop_last=True)) == 3 + 2


def test_unshuffled_keeps_index_order() -> None:
    batches = list(_sampler(shuffle=False))

    assert batches == [[0, 1], [2, 3], [4, 5], [6], [7, 8], [9, 10], [11]]


def test_lightning_injects_distributed_sampler_without_losing_groups() -> None:
    dataset = list(range(sum(GROUPS)))
    with _replace_dunder_methods(DataLoader, "dataset"), _replace_dunder_methods(BatchSampler):
        dataloader = DataLoader(
            dataset,
            batch_sampler=GroupedSkipBatchSampler(
                RandomSampler(dataset, generator=torch.Generator().manual_seed(38)),
                batch_size=2,
                first_batch_idx=1,
                group_sizes=GROUPS,
                seed=38,
            ),
        )

    injected = _update_dataloader(dataloader, DistributedSampler(dataset, num_replicas=2, rank=1, seed=38))

    batch_sampler = injected.batch_sampler
    assert isinstance(batch_sampler, GroupedSkipBatchSampler)
    assert (batch_sampler.group_sizes, batch_sampler.first_batch_idx, batch_sampler.seed) == (GROUPS, 1, 38)
    assert list(batch_sampler) == list(_sampler(_Distributed(2, 1)))[1:]
