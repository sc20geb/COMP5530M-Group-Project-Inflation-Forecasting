from copy import deepcopy
import torch


class EarlyStopping:

    def __init__(self, tol=1e-3, patientce=15):

        self.tol = tol
        self.patientce = patientce
        self.counter = 0
        self.bestModel = None
        self.bestLoss = None

    def __call__(self, model, validLoss):

        if self.bestLoss is None:
            self.bestLoss = validLoss
            self.bestModel = deepcopy(model.state_dict())

        elif self.tol <= self.bestLoss - validLoss:
            self.counter = 0
            self.bestLoss = validLoss
            self.bestModel = deepcopy(model.state_dict())

        else:
            self.counter += 1

            if self.counter == self.patientce:
                print(f"Early stopping. counter: {self.counter}")

                return True

        return False

    def restoreBestWeights(self, model):

        model.load_state_dict(self.bestModel)
        print("Best weights restored.")
