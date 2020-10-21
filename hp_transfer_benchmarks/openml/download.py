import logging

from hpolib.benchmarks.surrogates.exploring_openml import SVM
from hpolib.benchmarks.surrogates.exploring_openml import XGBoost
from hpolib.benchmarks.surrogates.exploring_openml import all_datasets


logger = logging.getLogger(__name__)


def _download_data_for_dataset(dataset_id, seed):
    SVM(dataset_id=dataset_id, rng=seed)
    XGBoost(dataset_id=dataset_id, rng=seed)


def download_all_data(dataset_ids, seed):
    for i, dataset_id in enumerate(dataset_ids, 1):
        logger.info(f"Loading dataset with id {dataset_id} ({i}/{len(dataset_ids)})")
        try:
            _download_data_for_dataset(dataset_id, seed)
        except KeyboardInterrupt as e:
            raise e
        except Exception:  # pylint: disable=broad-except
            logger.exception("Failed to load dataset")


if __name__ == "__main__":
    import argparse
    from hp_transfer_benchmarks.openml import OpenMLBenchmark

    parser = argparse.ArgumentParser("Download openml data")
    parser.add_argument("--overwrite_qualities", action="store_true")
    parser.add_argument("--dataset_id", default=None, type=int)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.all:
        dataset_ids = all_datasets
    elif args.dataset_id is not None:
        dataset_ids = [args.dataset_id]
    else:
        dataset_ids = OpenMLBenchmark.task_identifiers[:4]

    download_all_data(dataset_ids, OpenMLBenchmark.seed)
