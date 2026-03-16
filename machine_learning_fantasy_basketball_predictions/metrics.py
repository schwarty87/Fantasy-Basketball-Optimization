import torch
import torch.nn as nn
from typing import Optional

from constants import IGNORE_VAL
from machine_learning.metrics.abstract_metrics import AbstractMetric
from machine_learning.registries import metrics_registry
from machine_learning_fantasy_basketball_predictions.params.metrics_params import PairwiseRankingLossParams, BradleyTerryLossParams


@metrics_registry.register
class OptimizeTeamsLoss(AbstractMetric):
    def _error_unreduced(self, y_true, y_pred):
        remove = torch.all(y_pred == 0, dim=[1, 2])
        y_pred[remove] = torch.nan
        return - torch.sum(y_true * y_pred, dim=[1, 2])


@metrics_registry.register
class PairwiseRankingLoss(AbstractMetric):
    def _get_pairwise_weights(self, true_diffs: torch.Tensor, salary: torch.Tensor) -> torch.Tensor:
        """
        Calculates the weight matrix based on static features (Salary, Position, etc.)
        """
        # Default weights = 1.0
        weights = torch.ones_like(true_diffs)

        # Expand to (Batch, N, N)
        sal_i = salary.unsqueeze(2).expand_as(true_diffs)
        sal_j = salary.unsqueeze(1).expand_as(true_diffs)
        salary_gap = sal_i - sal_j

        # Upset (Winner is Cheaper)
        upset_mask = ((salary_gap < 0) & (true_diffs > 0)) | ((salary_gap > 0) & (true_diffs < 0))
        weights[upset_mask] = 1.0 + (0.1 * torch.abs(salary_gap[upset_mask]))

        # # Salary matches y_true (Similar Salaries)
        # expected_mask = ((salary_gap > 0) & (true_diffs > 0)) | ((salary_gap < 0) & (true_diffs < 0))
        # weights[expected_mask] = 1.0 / (1.0 + 0.1 * torch.abs(salary_gap[expected_mask]))

        close_threshold = 3.0
        close_mask = torch.abs(salary_gap) < close_threshold
        weights[close_mask] = weights[close_mask] * 2.0

        return weights

    def _error_unreduced(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Calculates the pairwise loss matrix.
        Returns a tensor of shape (Batch, N, N).
        Invalid pairs (where True[i] <= True[j]) are set to NaN.
        """
        salary = None
        if y_true.ndim == y_pred.ndim + 1:
            salary = y_true[..., 1]
            salary = torch.where(salary != IGNORE_VAL, salary, float('nan'))
            y_true = y_true[..., 0]

            if y_true.shape != y_pred.shape:
                raise ValueError('y_true and y_pred have to have the same shape!')

        # Ensure inputs are (Batch, N)
        keep_mask = torch.ones_like(y_true)
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)
            keep_mask = keep_mask.unsqueeze(0)
            if salary is not None:
                salary = salary.unsqueeze(0)
        elif y_pred.dim() == 3:
            # if there are 3 dimensions, we assume the dimensions are:
            # (num_days, num_games_per_day, num_players_per_game)
            # and we treat each sample (batch_size) as a single day
            keep_mask = torch.any(y_true != 0, dim=[2], keepdim=True).repeat(1, 1, y_true.shape[-1])

            # reshape
            y_pred = y_pred.view(y_pred.shape[0], -1)
            y_true = y_true.view(y_true.shape[0], -1)
            keep_mask = keep_mask.view(keep_mask.shape[0], -1)
            if salary is not None:
                salary = salary.view(salary.shape[0], -1)

        # 1. Expand to (Batch, N, N) for broadcasting
        # true_diffs[b, i, j] = true[b, i] - true[b, j]
        true_diffs = y_true.unsqueeze(2) - y_true.unsqueeze(1)

        # 2. Identify "Structural" Valid Pairs
        # We only define error on pairs where Player I > Player J
        # If Player I <= Player J, we shouldn't punish the model for predicting I < J.
        # We handle this by setting those cells to NaN (AbstractMetric ignores NaNs).
        structural_mask = (true_diffs > 0) & keep_mask.unsqueeze(1) & keep_mask.unsqueeze(2)

        # 3. Prepare Predictions for MarginRankingLoss
        # pred[i] vs pred[j]
        pred_i = y_pred.unsqueeze(2).expand_as(true_diffs)
        pred_j = y_pred.unsqueeze(1).expand_as(true_diffs)

        target = torch.where(true_diffs > 0, 1, -1)

        # 4. Calculate Raw Loss
        # We calculate loss for ALL pairs first, then mask.
        # (Computationally cheaper than boolean indexing for dense matrices on GPU)
        margin = self.params.margin
        if self.params.margin_is_pct_of_true_diffs:
            margin = margin * torch.abs(true_diffs)
        loss_matrix = torch.relu(margin - target * (pred_i - pred_j))

        # 5. Apply True Difference Weights (Optional)
        if self.params.use_true_diff_weights:
            # Scale loss by the size of the mistake (the true gap)
            loss_matrix = loss_matrix * torch.log1p(torch.abs(true_diffs))

        # 6. Apply Structural Mask (Set invalid pairs to NaN)
        # We use clone() to ensure we don't modify gradients in place unexpectedly
        loss_matrix = torch.where(structural_mask, loss_matrix, torch.tensor(float('nan'), device=loss_matrix.device))

        # # 2. Static Weights (Salary/Position) - Calculated Helper
        # if salary is not None:
        #     static_weights = self._get_pairwise_weights(true_diffs, salary)
        #     loss_matrix = loss_matrix * static_weights

        return loss_matrix

    @staticmethod
    def _apply_weights(errors: torch.Tensor, sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Override to handle broadcasting of weights from (Batch, N) to (Batch, N, N).
        Weight(i, j) = sqrt(Weight(i) * Weight(j))
        """
        if sample_weights is None:
            return errors

        # Ensure weights are at least (Batch, N)
        if sample_weights.dim() == 1:
            sample_weights = sample_weights.unsqueeze(0)

        # Calculate Pairwise Weights: (Batch, N, N)
        w_i = sample_weights.unsqueeze(2)
        w_j = sample_weights.unsqueeze(1)

        # Geometric mean seems appropriate (if either player is 0-weight, pair is 0-weight)
        pair_weights = (w_i * w_j).sqrt()

        # Broadcast pair_weights to errors shape if necessary
        # (Though errors should already be Batch, N, N)
        return errors * pair_weights

    @staticmethod
    def _mask_errors(errors: torch.Tensor, keep_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Override to handle broadcasting of mask from (Batch, N) to (Batch, N, N).
        Valid(i, j) = Valid(i) AND Valid(j)
        """
        if keep_mask is None:
            return errors

        keep_mask = PairwiseRankingLoss._reshape_keep_mask(keep_mask)

        # Apply mask: Set invalid player pairs to NaN
        # Note: 'errors' already has NaNs from the structural mask (i <= j).
        # This adds NaNs for invalid players.
        return torch.where(keep_mask, errors, torch.tensor(float('nan'), device=errors.device))

    @staticmethod
    def _reshape_keep_mask(keep_mask: torch.Tensor) -> torch.Tensor:
        keep_mask = keep_mask.view(keep_mask.shape[0], -1)

        # Create Pairwise Mask: (Batch, N, N)
        # Pair is valid only if BOTH players are valid
        m_i = keep_mask.unsqueeze(2).bool()
        m_j = keep_mask.unsqueeze(1).bool()
        return m_i & m_j


@metrics_registry.register
class BradleyTerryLoss(AbstractMetric):
    def __init__(self, params: BradleyTerryLossParams):
        super().__init__(params=params)
        self._loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

    def _error_unreduced(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        salary = None
        if y_true.ndim == y_pred.ndim + 1:
            salary = y_true[..., 1]
            salary = torch.where(salary != IGNORE_VAL, salary, float('nan'))
            y_true = y_true[..., 0]

            if y_true.shape != y_pred.shape:
                raise ValueError('y_true and y_pred have to have the same shape!')

        # Ensure inputs are (Batch, N)
        keep_mask = torch.ones_like(y_true)
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)
            keep_mask = keep_mask.unsqueeze(0)
            if salary is not None:
                salary = salary.unsqueeze(0)
        elif y_pred.dim() == 3:
            # if there are 3 dimensions, we assume the dimensions are:
            # (num_days, num_games_per_day, num_players_per_game)
            # and we treat each sample (batch_size) as a single day
            keep_mask = torch.any(y_true != 0, dim=[2], keepdim=True).repeat(1, 1, y_true.shape[-1])

            # reshape
            y_pred = y_pred.view(y_pred.shape[0], -1)
            y_true = y_true.view(y_true.shape[0], -1)
            keep_mask = keep_mask.view(keep_mask.shape[0], -1)
            if salary is not None:
                salary = salary.view(salary.shape[0], -1)

        # 1. Expand to (Batch, N, N) for broadcasting
        # true_diffs[b, i, j] = true[b, i] - true[b, j]
        true_diffs = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        pred_diffs = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)

        true_diffs_binary = torch.where(true_diffs > 0, 1, 0).to(dtype=pred_diffs.dtype)
        if self.params.margin:
            if self.params.margin_is_pct_of_true_diffs:
                ratio1 = y_true.unsqueeze(2) / y_true.unsqueeze(1)
                ratio2 = 1 / ratio1
                raise ValueError('we havent implemented this yet')
            else:
                mask = torch.abs(true_diffs) <= self.params.margin
            true_diffs_binary = torch.where(mask, 0.5, true_diffs_binary)
        return self._loss_fn(pred_diffs, true_diffs_binary)

    @staticmethod
    def _apply_weights(errors: torch.Tensor, sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Override to handle broadcasting of weights from (Batch, N) to (Batch, N, N).
        Weight(i, j) = sqrt(Weight(i) * Weight(j))
        """
        if sample_weights is None:
            return errors

        # Ensure weights are at least (Batch, N)
        if sample_weights.dim() == 1:
            sample_weights = sample_weights.unsqueeze(0)

        # Calculate Pairwise Weights: (Batch, N, N)
        w_i = sample_weights.unsqueeze(2)
        w_j = sample_weights.unsqueeze(1)

        # Geometric mean seems appropriate (if either player is 0-weight, pair is 0-weight)
        pair_weights = (w_i * w_j).sqrt()

        # Broadcast pair_weights to errors shape if necessary
        # (Though errors should already be Batch, N, N)
        return errors * pair_weights

    @staticmethod
    def _mask_errors(errors: torch.Tensor, keep_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Override to handle broadcasting of mask from (Batch, N) to (Batch, N, N).
        Valid(i, j) = Valid(i) AND Valid(j)
        """
        if keep_mask is None:
            return errors

        keep_mask = BradleyTerryLoss._reshape_keep_mask(keep_mask)

        # Apply mask: Set invalid player pairs to NaN
        # Note: 'errors' already has NaNs from the structural mask (i <= j).
        # This adds NaNs for invalid players.
        return torch.where(keep_mask, errors, torch.tensor(float('nan'), device=errors.device))

    @staticmethod
    def _reshape_keep_mask(keep_mask: torch.Tensor) -> torch.Tensor:
        keep_mask = keep_mask.view(keep_mask.shape[0], -1)

        # Create Pairwise Mask: (Batch, N, N)
        # Pair is valid only if BOTH players are valid
        m_i = keep_mask.unsqueeze(2).bool()
        m_j = keep_mask.unsqueeze(1).bool()
        return m_i & m_j
