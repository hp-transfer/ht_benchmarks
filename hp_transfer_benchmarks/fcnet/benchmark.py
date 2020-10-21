"""
Adapted from https://github.com/automl/nas_benchmarks
"""
import json
import os

from pathlib import Path

import ConfigSpace as CS
import h5py
import numpy as np

from hp_transfer_benchmarks.benchmark_settings import SingleTaskSetting
from hp_transfer_benchmarks.fcnet._configspaces import FCNetConfigspaceA
from hp_transfer_benchmarks.fcnet._configspaces import FCNetConfigspaceB
from hp_transfer_benchmarks.task import Task


def _fill_config(config_dict):
    # BOHB does not work with ordinals, so we encode hyperparameters with integers
    for hyperparameter, value in config_dict.items():
        if hyperparameter.startswith("batch"):
            value_map = [8, 16, 32, 64]
        elif hyperparameter.startswith("dropout"):
            value_map = [0.0, 0.3, 0.6]
        elif hyperparameter.startswith("n_unit"):
            value_map = [16, 32, 64, 128, 256, 512]
        elif hyperparameter.startswith("init_lr"):
            value_map = [5 * 1e-4, 1e-3, 5 * 1e-3, 1e-2, 5 * 1e-2, 1e-1]
        else:
            continue
        config_dict[hyperparameter] = value_map[value]
    return config_dict


class _FCNetCoreBenchmark:
    def __init__(self, path, dataset="fcnet_protein_structure_data.hdf5"):
        self.data = h5py.File(os.path.join(path, dataset), "r")

    def objective_function_deterministic(self, config, budget=100, index=0):
        assert 0 < budget <= 100  # check whether budget is in the correct bounds

        if isinstance(config, CS.Configuration):
            k = json.dumps(config.get_dictionary(), sort_keys=True)
        else:
            k = json.dumps(config, sort_keys=True)

        valid = self.data[k]["valid_mse"][index]
        return float(valid[budget - 1])

    def objective_function_test(self, config):
        if isinstance(config, CS.Configuration):
            k = json.dumps(config.get_dictionary(), sort_keys=True)
        else:
            k = json.dumps(config, sort_keys=True)
        return float(np.mean(self.data[k]["final_test_error"]))


def _get_core_benchmark(data_dir, dataset):
    if dataset == "slice":
        data_file = "fcnet_slice_localization_data.hdf5"
    elif dataset == "protein":
        data_file = "fcnet_protein_structure_data.hdf5"
    elif dataset == "naval":
        data_file = "fcnet_naval_propulsion_data.hdf5"
    elif dataset == "parkinson":
        data_file = "fcnet_parkinsons_telemonitoring_data.hdf5"
    else:
        raise ValueError
    return _FCNetCoreBenchmark(data_dir, data_file)


class FCNetBenchmark:
    task_identifiers = trajectory_ids = ["slice", "protein", "naval", "parkinson"]
    adjustment_ids = ["a", "b", "c"]

    def __init__(
        self, trajectory_id="protein", data_path="data/fcnet", adjustment_id="a"
    ):
        self.adjustment_id = adjustment_id
        self.data_path = Path(data_path) / "fcnet_tabular_benchmarks"
        self.dev_task_identifiers = [trajectory_id]
        self.eval_task_identifiers = [
            id_ for id_ in self.task_identifiers if id_ != trajectory_id
        ]

        n_dev_stages = FCNetConfigspaceA.n_development_stages
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

    def get_task_from_identifier(self, identifier, development_stage=None):
        benchmark = _get_core_benchmark(self.data_path, identifier)
        configspace = self.development_stage_to_configspace(development_stage)
        default_values = configspace.default_values()

        def evaluate_fn(config):
            config = {**default_values, **config}
            config = _fill_config(config)
            if "lr_schedule" in config.keys() and config["lr_schedule"] == "cosine2":
                config["lr_schedule"] = "cosine"
            return dict(
                loss=benchmark.objective_function_deterministic(
                    config, budget=configspace.budget
                ),
                info=dict(test_loss=benchmark.objective_function_test(config)),
            )

        representation = []
        return Task(
            evaluate_fn=evaluate_fn, representation=representation, identifier=identifier,
        )

    def development_stage_to_configspace(self, development_stage):
        if self.adjustment_id == "a":
            return FCNetConfigspaceA(development_stage)
        elif self.adjustment_id == "b":
            return FCNetConfigspaceB(development_stage)
        else:
            raise ValueError
