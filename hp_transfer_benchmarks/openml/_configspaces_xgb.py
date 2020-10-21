import ConfigSpace as CS

from ConfigSpace import hyperparameters as CSH


class XGBoostConfigSpaceA(CS.ConfigurationSpace):
    n_development_stages = 2

    def __init__(self, development_stage=-1):
        super().__init__()

        self.development_stage = development_stage

        nrounds = CSH.UniformIntegerHyperparameter("nrounds", lower=1, upper=5000)
        subsample = CSH.UniformFloatHyperparameter("subsample", lower=0, upper=1)
        eta = CSH.UniformFloatHyperparameter(
            "eta", lower=2 ** -10, upper=2 ** 0, log=True,
        )
        lambda_ = CSH.UniformFloatHyperparameter(
            "lambda", lower=2 ** -10, upper=2 ** 10, log=True,
        )
        alpha = CSH.UniformFloatHyperparameter(
            "alpha", lower=2 ** -10, upper=2 ** 10, log=True,
        )
        colsample_bytree = CSH.UniformFloatHyperparameter(
            "colsample_bytree", lower=0, upper=1,
        )
        colsample_bylevel = CSH.UniformFloatHyperparameter(
            "colsample_bylevel", lower=0, upper=1,
        )
        min_child_weight = CSH.UniformFloatHyperparameter(
            "min_child_weight", lower=2 ** 0, upper=2 ** 7, log=True,
        )
        max_depth = CSH.UniformIntegerHyperparameter("max_depth", lower=1, upper=15)

        if development_stage == 0:
            self.add_hyperparameters([nrounds, subsample, eta, lambda_, alpha])
        elif development_stage == 1 or development_stage == -1:
            self.add_hyperparameters(
                [
                    nrounds,
                    subsample,
                    eta,
                    lambda_,
                    alpha,
                    colsample_bytree,
                    colsample_bylevel,
                    max_depth,
                    min_child_weight,
                ]
            )
        else:
            ValueError()

    def default_values(self):
        if self.development_stage == 0:
            return {
                "colsample_bytree": 1,
                "colsample_bylevel": 1,
                "max_depth": 6,
                "min_child_weight": 1,
                "booster": "gbtree",
            }
        elif self.development_stage == 1 or self.development_stage == -1:
            return {"booster": "gbtree"}
        else:
            ValueError()


class XGBoostConfigSpaceB(CS.ConfigurationSpace):
    n_development_stages = 2

    def __init__(self, development_stage=-1):
        super().__init__()

        self.development_stage = development_stage

        nrounds = CSH.UniformIntegerHyperparameter("nrounds", lower=1, upper=5000)
        subsample = CSH.UniformFloatHyperparameter("subsample", lower=0, upper=1)
        eta = CSH.UniformFloatHyperparameter(
            "eta", lower=2 ** -10, upper=2 ** 0, log=True,
        )
        lambda_ = CSH.UniformFloatHyperparameter(
            "lambda", lower=2 ** -10, upper=2 ** 10, log=True,
        )
        alpha = CSH.UniformFloatHyperparameter(
            "alpha", lower=2 ** -10, upper=2 ** 10, log=True,
        )
        booster = CSH.CategoricalHyperparameter("booster", choices=["gblinear", "gbtree"])

        self.add_hyperparameters([booster, nrounds, subsample, eta, lambda_, alpha])

    def default_values(self):
        if self.development_stage == 0:
            return {
                "colsample_bytree": 1,
                "colsample_bylevel": 1,
                "max_depth": 6,
                "min_child_weight": 1,
            }
        elif self.development_stage == 1 or self.development_stage == -1:
            return {
                "colsample_bytree": 1,
                "colsample_bylevel": 0.5,
                "max_depth": 10,
                "min_child_weight": 10,
            }
        else:
            ValueError()
