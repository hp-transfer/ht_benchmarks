import logging
import random

import ConfigSpace as CS

from hp_transfer_benchmarks.benchmark_settings import SingleTaskSetting
from hp_transfer_benchmarks.openml._configspaces_svm import SVMConfigSpaceA
from hp_transfer_benchmarks.openml._configspaces_svm import SVMConfigSpaceB
from hp_transfer_benchmarks.openml._configspaces_xgb import XGBoostConfigSpaceA
from hp_transfer_benchmarks.openml._configspaces_xgb import XGBoostConfigSpaceB
from hp_transfer_benchmarks.openml._surrogates_hpolib import SVM
from hp_transfer_benchmarks.openml._surrogates_hpolib import XGBoost
from hp_transfer_benchmarks.openml._surrogates_hpolib import all_datasets as _DATASET_IDS
from hp_transfer_benchmarks.task import Task


logger = logging.getLogger(__name__)
logging.getLogger("openml").setLevel(logging.WARNING)


def _get_core_benchmark(dataset_id, algorithm, seed):
    if algorithm == "svm":
        return SVM(dataset_id=dataset_id, rng=seed)
    elif algorithm == "xgb":
        return XGBoost(dataset_id=dataset_id, rng=seed)
    else:
        raise ValueError()


def _deterministic_shuffle_with_return(iter_, seed):
    return random.Random(seed).sample(iter_, len(iter_))


class OpenMLBenchmark:
    seed = 1
    task_identifiers = _deterministic_shuffle_with_return(_DATASET_IDS, seed)
    adjustment_ids = ["a", "b", "c"]

    def __init__(
        self, trajectory_id=0, num_eval_tasks=1, algorithm="svm", adjustment_id="a"
    ):
        self.adjustment_id = adjustment_id
        self._algorithm = algorithm

        self.dev_task_identifiers = [self.task_identifiers[trajectory_id]]
        self.eval_task_identifiers = self.task_identifiers[-num_eval_tasks:]

        n_dev_stages = self._get_meta_configspace().n_development_stages
        self._setting = SingleTaskSetting(
            benchmark=self, dev_stages=list(range(n_dev_stages))
        )

    @property
    def eval_batch(self):
        return self._setting.eval_batch

    @property
    def dev_trajectory(self):
        return self._setting.dev_trajectory

    @property
    def configspace_trajectory(self):
        return self._setting.configspace_trajectory

    def get_task_from_identifier(
        self, identifier, development_stage=None,  # pylint: disable=unused-argument
    ):
        benchmark = _get_core_benchmark(identifier, self._algorithm, self.seed)

        configspace = self.development_stage_to_configspace(development_stage)
        default_values = configspace.default_values()

        def evaluate_fn(config):
            config = {**default_values, **config}  # Config overrides defaults
            config = CS.util.deactivate_inactive_hyperparameters(
                configuration_space=benchmark.get_configuration_space(),
                configuration=config,
            )
            return dict(
                loss=benchmark.objective_function(config)["function_value"], info=dict()
            )

        return Task(
            evaluate_fn=evaluate_fn, representation=identifier, identifier=identifier,
        )

    def _get_meta_configspace(self):
        if self._algorithm == "svm":
            if self.adjustment_id == "a":
                return SVMConfigSpaceA
            elif self.adjustment_id == "b":
                return SVMConfigSpaceB
            else:
                raise ValueError()
        elif self._algorithm == "xgb":
            if self.adjustment_id == "a":
                return XGBoostConfigSpaceA
            elif self.adjustment_id == "b":
                return XGBoostConfigSpaceB
            else:
                raise ValueError()
        else:
            raise ValueError()

    def development_stage_to_configspace(self, development_stage):
        return self._get_meta_configspace()(development_stage)
