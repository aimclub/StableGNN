from stable_gnn.visualization.contracts.strength_constructor_contract import StrengthConstructorContract
from stable_gnn.visualization.utils.fill_strength import fill_strength
from stable_gnn.visualization.config.parameters.defaults import Defaults


class StrengthConstructor:

    def __call__(self, contract: StrengthConstructorContract) -> tuple:
        _push_vertex_strength = Defaults.push_vertex_strength
        _push_edge_strength = Defaults.push_edge_strength
        _pull_edge_strength = Defaults.pull_edge_strength
        _pull_center_strength = Defaults.pull_center_strength

        push_vertex_strength = fill_strength(contract.push_vertex_strength,
                                             _push_vertex_strength)
        push_edge_strength = fill_strength(contract.push_edge_strength,
                                           _push_edge_strength)
        pull_edge_strength = fill_strength(contract.pull_edge_strength,
                                           _pull_edge_strength)
        pull_center_strength = fill_strength(contract.pull_center_strength,
                                             _pull_center_strength)

        return push_vertex_strength, push_edge_strength, pull_edge_strength, pull_center_strength
