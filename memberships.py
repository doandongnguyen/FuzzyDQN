import numpy as np
from fuzzy import FIS


# Define Membership for MountainCars
class BuildFis:
    def __init__(self):
        p = FIS.InputStateVariable(FIS.Trapeziums(-1.2, -1.2, -1.2, -0.775),
                                   FIS.Trapeziums(-1.2, -0.775, -0.775, -0.35),
                                   FIS.Trapeziums(-0.775, -0.35, -0.35, 0.075),
                                   FIS.Trapeziums(-0.35, 0.075, 0.075, 0.5),
                                   FIS.Trapeziums(0.075, 0.5, 0.5, 0.5))
        v = FIS.InputStateVariable(FIS.Trapeziums(-0.07, -0.07, -0.07, -0.035),
                                   FIS.Trapeziums(-0.07, -0.035, -0.035, 0.),
                                   FIS.Trapeziums(-0.035, 0., 0., 0.035),
                                   FIS.Trapeziums(0., 0.035, 0.035, 0.07),
                                   FIS.Trapeziums(0.035, 0.035, 0.035, 0.07))
        self.rules = FIS.Rules(p, v)
        self.fis = FIS.FIS(Rules=self.rules)

    def get_truth_values(self, state):
        return np.asarray(self.fis.truth_values(state))

    def shape(self):
        return self.rules.get_number_of_rules(),
