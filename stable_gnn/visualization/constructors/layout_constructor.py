import numpy as np

from stable_gnn.visualization.config.parameters.defaults import Defaults
from stable_gnn.visualization.contracts.core_model_contract import CoreModelContract
from stable_gnn.visualization.contracts.layout_contract import LayoutContract
from stable_gnn.visualization.equations.core_physical_model import CorePhysicalModel
from stable_gnn.visualization.equations.edge_list_to_incidence_matrix import edge_list_to_incidence_matrix
from stable_gnn.visualization.equations.init_position import init_position
from stable_gnn.visualization.exceptions.exceptions_classes import ParamsValidationException


class LayoutConstructor:

    def __call__(self, contract: LayoutContract):
        vertex_coord = init_position(contract.vertex_num, scale=Defaults.layout_scale_initial)

        self._validate(vertex_coord)

        centers = [np.array([0, 0])]

        core_model_contract: CoreModelContract = CoreModelContract(
            nums=contract.vertex_num,
            forces={
                Defaults.node_attraction_key: contract.pull_edge_strength,
                Defaults.node_repulsion_key: contract.push_vertex_strength,
                Defaults.edge_repulsion_key: contract.push_edge_strength,
                Defaults.center_of_gravity_key: contract.pull_center_strength,
            },
            centers=centers,
        )
        model: CorePhysicalModel = CorePhysicalModel(core_model_contract)

        vertex_coord = model.build(vertex_coord, edge_list_to_incidence_matrix(contract.vertex_num, contract.edge_list))
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
