from SHAP_Plots.common import Model
from SHAP_Plots.common import Instance
from SHAP_Plots.datatypes import Data
from SHAP_Plots.links import Link

class Explanation:
    def __init__(self):
        pass


class AdditiveExplanation(Explanation):
    def __init__(self, base_value, out_value, effects, effects_var, instance, link, model, data):
        self.base_value = base_value
        self.out_value = out_value
        self.effects = effects
        self.effects_var = effects_var
        # assert isinstance(instance, Instance)
        self.instance = instance
        # assert isinstance(link, Link)
        self.link = link
        # assert isinstance(model, Model)
        self.model = model
        # assert isinstance(data, Data)
        self.data = data

    # def _rdepr_pretty_(self, pp, cycle):
    #     print(pp)
    #     return visualizers.visualize(self)
