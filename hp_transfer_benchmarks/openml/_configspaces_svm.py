import ConfigSpace as CS

from ConfigSpace import hyperparameters as CSH


class SVMConfigSpaceA(CS.ConfigurationSpace):
    n_development_stages = 2

    def __init__(self, development_stage=-1):
        super().__init__()

        self.development_stage = development_stage

        cost = CSH.UniformFloatHyperparameter("cost", 2 ** -10, 2 ** 10, log=True)
        gamma = CSH.UniformFloatHyperparameter("gamma", 2 ** -5, 2 ** 5, log=True)
        degree = CSH.UniformIntegerHyperparameter("degree", 2, 5)

        if development_stage == 0:
            self.add_hyperparameters([cost, gamma])
        elif development_stage == 1 or development_stage == -1:
            self.add_hyperparameters([cost, degree])
        else:
            ValueError()

    def default_values(self):
        if self.development_stage == 0:
            return {"kernel": "radial", "degree": 3}
        elif self.development_stage == 1 or self.development_stage == -1:
            return {"kernel": "polynomial", "gamma": 1}
        else:
            ValueError()


class SVMConfigSpaceB(CS.ConfigurationSpace):
    n_development_stages = 2

    def __init__(self, development_stage=-1):
        super().__init__()

        self.development_stage = development_stage

        kernel = CSH.CategoricalHyperparameter(
            "kernel", choices=["linear", "polynomial", "radial"],
        )

        if development_stage == 0:
            cost = CSH.UniformFloatHyperparameter("cost", 2 ** -5, 2 ** 5, log=True)
            self.add_hyperparameters([cost, kernel])
        elif development_stage == 1 or development_stage == -1:
            cost = CSH.UniformFloatHyperparameter("cost", 2 ** -10, 2 ** 10, log=True)
            self.add_hyperparameters([cost, kernel])
        else:
            ValueError()

    def default_values(self):
        if self.development_stage == 0:
            return {"gamma": 1, "degree": 5}
        elif self.development_stage == 1 or self.development_stage == -1:
            return {"gamma": 1, "degree": 5}
        else:
            ValueError()
