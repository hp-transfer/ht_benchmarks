import ConfigSpace as CS


class NASConfigspaceA(CS.ConfigurationSpace):
    n_development_stages = 2

    def __init__(self, development_stage=-1):
        super().__init__()

        self.budget = 100

        self.development_stage = development_stage

        if development_stage == 0:
            ops = ["none", "skip_connect", "nor_conv_3x3", "avg_pool_3x3"]
        else:
            ops = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]

        self.add_hyperparameter(CS.CategoricalHyperparameter("1<-0", ops))
        self.add_hyperparameter(CS.CategoricalHyperparameter("2<-0", ops))
        self.add_hyperparameter(CS.CategoricalHyperparameter("3<-0", ops))
        self.add_hyperparameter(CS.CategoricalHyperparameter("3<-1", ops))
        self.add_hyperparameter(CS.CategoricalHyperparameter("2<-1", ops))
        self.add_hyperparameter(CS.CategoricalHyperparameter("3<-2", ops))

    def default_values(self):  # pylint: disable=no-self-use
        return {}


class NASConfigspaceB(CS.ConfigurationSpace):
    n_development_stages = 2

    def __init__(self, development_stage=-1):
        super().__init__()

        self.budget = 100

        self.development_stage = development_stage

        ops = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]

        self.add_hyperparameter(CS.CategoricalHyperparameter("2<-0", ops))
        self.add_hyperparameter(CS.CategoricalHyperparameter("3<-0", ops))
        self.add_hyperparameter(CS.CategoricalHyperparameter("3<-2", ops))
        if self.development_stage == 1:
            self.add_hyperparameter(CS.CategoricalHyperparameter("3<-1", ops))
            self.add_hyperparameter(CS.CategoricalHyperparameter("2<-1", ops))
            self.add_hyperparameter(CS.CategoricalHyperparameter("1<-0", ops))

    def default_values(self):
        if self.development_stage == 0:
            return {
                "3<-1": "none",
                "2<-1": "none",
                "1<-0": "none",
            }
        else:
            return {}
