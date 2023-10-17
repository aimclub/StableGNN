import numpy as np

from stable_gnn.visualization.contracts.layout_contract import LayoutContract
from stable_gnn.visualization.config.parameters.defaults import Defaults
from stable_gnn.visualization.equations.edge_list_to_incidence_matrix import edge_list_to_incidence_matrix
from stable_gnn.visualization.equations.init_position import init_position
from stable_gnn.visualization.exceptions.exceptions_classes import ParamsValidationException


class LayoutConstructor:

    def __call__(self, contract: LayoutContract):
        vertex_coord = init_position(contract.vertex_num, scale=Defaults.layout_scale_initial)

        self._validate(vertex_coord)

        centers = [np.array([0, 0])]

        sim = Simulator(
            nums=contract.vertex_num,
            forces={
                Simulator.NODE_ATTRACTION: contract.pull_edge_strength,
                Simulator.NODE_REPULSION: contract.push_vertex_strength,
                Simulator.EDGE_REPULSION: contract.push_edge_strength,
                Simulator.CENTER_GRAVITY: contract.pull_center_strength,
            },
            centers=centers,
        )

        vertex_coord = sim.simulate(vertex_coord,
                                    edge_list_to_incidence_matrix(contract.vertex_num,
                                                                  contract.edge_list))
        vertex_coord = ((vertex_coord - vertex_coord.min(0)) /
                        (vertex_coord.max(0) - vertex_coord.min(0)) *
                        Defaults.vertex_coord_multiplier + Defaults.vertex_coord_modifier)

        return vertex_coord

    @staticmethod
    def _validate(vertex_coord):
        is_valid = (vertex_coord.max() <= Defaults.vertex_coord_max and
                    vertex_coord.min() >= Defaults.vertex_coord_min)

        if not is_valid:
            raise ParamsValidationException
