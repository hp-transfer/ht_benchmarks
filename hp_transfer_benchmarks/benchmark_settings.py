import logging
import pprint


logger = logging.getLogger(__name__)


class BatchSampler:
    def __init__(self, batch_elements, transform_fn=None):
        self.transform_fn = transform_fn
        self.batch_elements = batch_elements

    def __iter__(self):
        for element in self.batch_elements:
            if self.transform_fn is None:
                yield element
            else:
                yield self.transform_fn(element)

    def __len__(self):
        return len(self.batch_elements)


class TrajectorySampler:
    def __init__(self, sequence, num_steps=None, batch_size=None, transform_fn=None):
        if num_steps is not None and batch_size is not None:
            raise ValueError()

        self.num_steps = num_steps
        self.batch_size = batch_size
        self.sequence = sequence
        self.transform_fn = transform_fn

    def __iter__(self):
        if self.num_steps is not None:
            # Repeat self.sequence in each step
            # E.g., sequence=[1, 2, 3] we get [1, 2, 3], [1, 2, 3], [1, 2, 3], ...
            batch_sampler = BatchSampler(self.sequence, self.transform_fn)
            for _ in range(self.num_steps):
                yield batch_sampler
        elif self.batch_size is not None:
            # Each step, yield a batch with corresponding value
            # E.g., sequence=[1, 2, 3] and batch_size=3 we get [1, 1, 1], [2, 2, 2], ...
            for element in self.sequence:
                yield BatchSampler([element] * self.batch_size, self.transform_fn)
        else:
            # Each step yield an element from sequence
            # E.g., sequence=[1, 2, 3] we get 1, 2, 3
            for element in self.sequence:
                if self.transform_fn is None:
                    yield element
                else:
                    yield self.transform_fn(element)

    def __len__(self):
        if self.num_steps is not None:
            return self.num_steps
        else:
            return len(self.sequence)


class SingleTaskSetting:
    def __init__(self, benchmark, dev_stages):
        self.dev_stages = dev_stages
        self.benchmark = benchmark

        eval_identifiers = self.benchmark.eval_task_identifiers
        dev_identifiers = self.benchmark.dev_task_identifiers
        logger.info(f"Using training identifier {pprint.pformat(dev_identifiers)}")
        logger.info(f"Using evaluation identifiers {pprint.pformat(eval_identifiers)}")

    @property
    def configspace_trajectory(self):
        return TrajectorySampler(
            self.dev_stages, transform_fn=self.benchmark.development_stage_to_configspace,
        )

    @property
    def eval_batch(self):
        return BatchSampler(
            self.benchmark.eval_task_identifiers,
            transform_fn=self.benchmark.get_task_from_identifier,
        )

    @property
    def dev_trajectory(self):
        return TrajectorySampler(
            self.benchmark.dev_task_identifiers,
            transform_fn=self.benchmark.get_task_from_identifier,
            num_steps=len(self.dev_stages),
        )
