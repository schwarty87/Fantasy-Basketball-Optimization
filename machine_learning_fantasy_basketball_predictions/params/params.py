import os
import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict
from utils import update_dir_name
from machine_learning.params.params import FeaturesParams, MachineLearningParams, TrainingParams, ArchitectureParams
from machine_learning.params.abstract_params import AbstractParams
from machine_learning.registries import params_registry


@params_registry.register
@dataclass(kw_only=True)
class FeaturesParamsFantasyBasketballPredictions(FeaturesParams):
    num_bench_players: int = 6
    last_month_in_train: bool = False
    sample_per_game: bool = True
    # If True, add dummy games for teams that didn't play, so that we have a sample for every team for every day
    # This is used whhen we want to have access to all games in a given day together
    all_games_every_day: bool = False
    extra_weight_on_certain_date: Optional[datetime.date] = None

    def __post_init__(self):
        super().__post_init__()
        if self.features_path is None:
            self.features_path = os.path.join(update_dir_name('features'), 'features.feather')

        self.all_feature_names_list = sorted(set(self.all_feature_names_list + ['EJECTED', 'LEFT_MIDGAME']))

        # adding columns needed for calculating the mean ratio
        for c in self.all_feature_names_list:
            if c.endswith('_mean_ratio'):
                base_name = c.replace('_mean_ratio', '')
                self.all_feature_names_list += [base_name, base_name + '_mean5']
        self.all_feature_names_list = sorted(set(self.all_feature_names_list))

        if self.sample_per_game and self.feature_type_to_input_shape is not None:
            num_players_per_game = 10 + 2 * self.num_bench_players
            for k in self.feature_type_to_input_shape.keys():
                self.feature_type_to_input_shape[k] = (num_players_per_game,) + (self.feature_type_to_input_shape[k])

    def to_dict(self, dict_factory=None) -> Dict:
        return_dict = super().to_dict()
        if self.extra_weight_on_certain_date is not None:
            return_dict['extra_weight_on_certain_date'] = str(self.extra_weight_on_certain_date)
        return return_dict

    @classmethod
    def init_from_dict(cls: dataclass, params_dict: Dict) -> AbstractParams:
        if params_dict.get('extra_weight_on_certain_date') is not None:
            params_dict['extra_weight_on_certain_date'] = \
                datetime.date.fromisoformat(params_dict['extra_weight_on_certain_date'])

        return super().init_from_dict(params_dict=params_dict)


@params_registry.register
@dataclass
class ArchitectureParamsFantasyBasketballPredictions(ArchitectureParams):
    pass


@params_registry.register
@dataclass
class TrainingParamsFantasyBasketballPredictions(TrainingParams):
    num_days_per_batch: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.batch_size and self.num_days_per_batch:
            raise ValueError('only one of batch_size and num_days_per_batch can be non-None!')


@params_registry.register
@dataclass
class MachineLearningParamsFantasyBasketballPredictions(MachineLearningParams):
    features_params: FeaturesParamsFantasyBasketballPredictions = field(default_factory=FeaturesParamsFantasyBasketballPredictions)
    training_params: TrainingParamsFantasyBasketballPredictions = field(default_factory=TrainingParamsFantasyBasketballPredictions)
    architecture_params: ArchitectureParamsFantasyBasketballPredictions = field(default_factory=ArchitectureParamsFantasyBasketballPredictions)
