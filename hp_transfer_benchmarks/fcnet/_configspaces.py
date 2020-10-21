import ConfigSpace as CS


class FCNetConfigspaceA(CS.ConfigurationSpace):
    n_development_stages = 2

    def __init__(self, development_stage=-1):
        super().__init__()

        self.development_stage = development_stage

        self.add_hyperparameter(
            CS.UniformIntegerHyperparameter("dropout_1", lower=0, upper=2)
        )
        self.add_hyperparameter(
            CS.UniformIntegerHyperparameter("dropout_2", lower=0, upper=2)
        )
        self.add_hyperparameter(
            CS.UniformIntegerHyperparameter("init_lr", lower=0, upper=5)
        )
        self.add_hyperparameter(
            CS.CategoricalHyperparameter("activation_fn_1", choices=["relu", "tanh"])
        )
        self.add_hyperparameter(
            CS.CategoricalHyperparameter("activation_fn_2", choices=["relu", "tanh"])
        )

        if development_stage == 0:
            self.budget = 50
            self.add_hyperparameter(
                CS.UniformIntegerHyperparameter("batch_size", lower=0, upper=3)
            )
        else:
            self.budget = 100

    def default_values(self):
        if self.development_stage == 0:
            return {"lr_schedule": "const", "n_units_1": 1, "n_units_2": 1}
        elif self.development_stage == 1 or self.development_stage == -1:
            return {
                "lr_schedule": "const",
                "n_units_1": 5,
                "n_units_2": 5,
                "batch_size": 1,
            }
        else:
            ValueError()


class FCNetConfigspaceB(CS.ConfigurationSpace):
    n_development_stages = 2

    def __init__(self, development_stage=-1):
        super().__init__()

        self.budget = 100

        self.development_stage = development_stage

        self.add_hyperparameter(
            CS.UniformIntegerHyperparameter("n_units_1", lower=0, upper=5)
        )
        self.add_hyperparameter(
            CS.UniformIntegerHyperparameter("n_units_2", lower=0, upper=5)
        )
        self.add_hyperparameter(
            CS.UniformIntegerHyperparameter("dropout_1", lower=0, upper=2)
        )
        self.add_hyperparameter(
            CS.UniformIntegerHyperparameter("dropout_2", lower=0, upper=2)
        )
        self.add_hyperparameter(
            CS.UniformIntegerHyperparameter("init_lr", lower=0, upper=5)
        )
        self.add_hyperparameter(
            CS.UniformIntegerHyperparameter("batch_size", lower=0, upper=3)
        )
        if self.development_stage == 1:
            self.add_hyperparameter(
                CS.CategoricalHyperparameter("activation_fn_1", choices=["relu", "tanh"])
            )
            self.add_hyperparameter(
                CS.CategoricalHyperparameter("activation_fn_2", choices=["relu", "tanh"])
            )

    def default_values(self):
        if self.development_stage == 0:
            return {
                "lr_schedule": "const",
                "activation_fn_1": "tanh",
                "activation_fn_2": "tanh",
            }
        elif self.development_stage == 1 or self.development_stage == -1:
            return {"lr_schedule": "cosine"}
        else:
            ValueError()
