from stable_gnn.visualization.config.parameters.defaults import Defaults
from stable_gnn.visualization.contracts.core_model_contract import CoreModelContract


class CorePhysicalModel:
    __node_attraction = Defaults.node_attraction_key
    __node_repulsion = Defaults.node_repulsion_key
    __edge_repulsion = Defaults.edge_repulsion_key
    __center_of_gravity = Defaults.center_of_gravity_key

    def __init__(self, contract: CoreModelContract):
        pass

    def simulate(self, init_position, H, max_iter=Defaults.max_iterations, epsilon=Defaults.epsilon,
                 delta=Defaults.delta):
        pass
