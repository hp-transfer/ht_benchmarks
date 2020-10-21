from hpolib.benchmarks.nas import nasbench_201

from hp_transfer_benchmarks.benchmark_settings import SingleTaskSetting
from hp_transfer_benchmarks.nas201._configspaces import NASConfigspaceA
from hp_transfer_benchmarks.nas201._configspaces import NASConfigspaceB
from hp_transfer_benchmarks.task import Task


def _get_core_benchmark(dataset_id):
    if dataset_id == "cifar10":
        return nasbench_201.Cifar10NasBench201Benchmark()
    elif dataset_id == "cifar100":
        return nasbench_201.Cifar100NasBench201Benchmark()
    elif dataset_id == "ImageNet16-120":
        return nasbench_201.ImageNetNasBench201Benchmark()
    else:
        raise ValueError()


class NASBenchmark:
    task_identifiers = trajectory_ids = ["cifar10", "cifar100", "ImageNet16-120"]
    adjustment_ids = ["a", "b"]

    def __init__(self, trajectory_id="cifar10", adjustment_id="a"):
        self.adjustment_id = adjustment_id
        self.dev_task_identifiers = [trajectory_id]
        self.eval_task_identifiers = [
            id_ for id_ in self.task_identifiers if id_ != trajectory_id
        ]

        n_dev_stages = NASConfigspaceA.n_development_stages
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
        benchmark = _get_core_benchmark(identifier)
        configspace = self.development_stage_to_configspace(development_stage)
        default_values = configspace.default_values()

        def evaluate_fn(config):
            config = {**default_values, **config}
            return dict(
                loss=benchmark.objective_function(config)["function_value"],
                info=dict(
                    test_loss=benchmark.objective_function_test(config)["function_value"]
                ),
            )

        representation = []
        return Task(
            evaluate_fn=evaluate_fn, representation=representation, identifier=identifier,
        )

    def development_stage_to_configspace(self, development_stage):
        if self.adjustment_id == "a":
            return NASConfigspaceA(development_stage)
        elif self.adjustment_id == "b":
            return NASConfigspaceB(development_stage)
        else:
            raise ValueError
