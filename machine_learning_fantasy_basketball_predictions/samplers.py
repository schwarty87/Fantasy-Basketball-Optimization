import numpy as np
from constants import NUM_TEAMS
from torch.utils.data import Sampler, Subset
from typing import List, Iterator
from collections import defaultdict
import logging


class ByDateBatchSampler(Sampler[List[int]]):
    """
    A PyTorch Sampler that creates batches containing all samples
    from a specified number of unique dates.

    It is designed to work with a torch.utils.data.Subset and a
    pre-computed map of original_index -> date.
    """

    def __init__(self,
                 subset_dataset: Subset,
                 original_index_to_date_map: np.ndarray,
                 num_days_per_batch: int,
                 shuffle: bool = False):
        super().__init__()

        self.num_days_per_batch = num_days_per_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed=42)

        # Build a map of {date -> list_of_SUBSET_indices}
        self.date_to_subset_indices = defaultdict(list)

        for subset_idx, original_idx in enumerate(subset_dataset.indices):
            # Find the date for this sample in the original dataset
            date = original_index_to_date_map[original_idx]
            # Map the date to the SUBSET index
            self.date_to_subset_indices[date].append(subset_idx)

        # # asserting we have the correct amount of indices per day
        # num_games_per_day = NUM_TEAMS // 2
        # for date, date_indices in self.date_to_subset_indices.items():
        #     if len(date_indices) != num_games_per_day:
        #         raise ValueError(f'Incorrect amount of games for date {date}! '
        #                          f'Should be {num_games_per_day}, '
        #                          f'but got {len(date_indices)}')

        self.unique_dates = list(self.date_to_subset_indices.keys())

        if not self.unique_dates:
            logging.warning("ByDateBatchSampler: No dates found for the given indices.")

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            # Shuffle the unique dates
            self.rng.shuffle(self.unique_dates)

        # Group dates into batches
        for i in range(0, len(self.unique_dates), self.num_days_per_batch):
            batch_dates = self.unique_dates[i:i + self.num_days_per_batch]

            # Collect all SUBSET indices for the dates in this batch
            batch_indices = []
            for date in batch_dates:
                batch_indices.extend(self.date_to_subset_indices[date])

            if batch_indices:
                yield batch_indices

    def __len__(self) -> int:
        # This returns the number of batches
        if not self.unique_dates:
            return 0
        return (len(self.unique_dates) + self.num_days_per_batch - 1) // self.num_days_per_batch