from copy import deepcopy

import numpy as np

from sklearn.metrics import euclidean_distances

from stable_gnn.visualization.config.parameters.defaults import Defaults
from stable_gnn.visualization.contracts.core_model_contract import CoreModelContract
from stable_gnn.visualization.equations.calc_edge_center import calc_edge_center
from stable_gnn.visualization.equations.calc_safe_div import safe_div


class CorePhysicalModel:
    __node_attraction_key = Defaults.node_attraction_key
    __node_repulsion_key = Defaults.node_repulsion_key
    __edge_repulsion_key = Defaults.edge_repulsion_key
    __center_of_gravity_key = Defaults.center_of_gravity_key

    def __init__(self, contract: CoreModelContract):

        self.contract = contract

        self.nums = self.__get_nums()

        self.node_attraction = contract.forces.get(self.__node_attraction_key, None)
        self.node_repulsion = contract.forces.get(self.__node_repulsion_key, None)
        self.edge_repulsion = contract.forces.get(self.__edge_repulsion_key, None)
        self.center_gravity = contract.forces.get(self.__center_of_gravity_key, None)

        self.num_centers = len(contract.centers)

        self.centers = contract.centers

        if self.node_repulsion is not None and isinstance(self.node_repulsion, float):
            self.node_repulsion = [self.node_repulsion] * self.num_centers

        if self.center_gravity is not None and isinstance(self.center_gravity, float):
            self.center_gravity = [self.center_gravity] * self.num_centers

        self.damping_factor = contract.damping_factor

    def __get_nums(self):
        return [self.contract.nums] if isinstance(self.contract.nums, int) else self.contract.nums

    def build(self,
              init_position,
              H,
              max_iter=Defaults.max_iterations,
              epsilon=Defaults.epsilon,
              delta=Defaults.delta):

        position = init_position.copy()
        velocity = np.zeros_like(position)
        damping = Defaults.damping

        for it in range(max_iter):
            position, velocity, stop_iterations = self.__make_one_step(position, velocity, H, epsilon, damping, delta)

            if stop_iterations:
                break

            damping *= self.damping_factor

        return position

    def __make_one_step(self, position, velocity, H, epsilon, damping, delta):
        edge_center = calc_edge_center(H, position)

        vertex_to_vertex_distance = euclidean_distances(position)
        vertex_to_edge_distance = euclidean_distances(position, edge_center) * H
        edge_to_edge_distance = euclidean_distances(edge_center)

        force = np.zeros_like(position)

        if self.node_attraction is not None:
            force_modifier = (self.__node_attraction(position, edge_center, vertex_to_edge_distance) *
                              self.node_attraction)
            force += force_modifier

        if self.node_repulsion is not None:
            force_modifier = self.__node_repulsion(position, vertex_to_vertex_distance)
            if self.num_centers == 1:
                force_modifier *= self.node_repulsion[0]
            else:
                masks = np.zeros((position.shape[0], 1))
                masks[:self.nums[0]] = self.node_repulsion[0]
                masks[self.nums[0]:] = self.node_repulsion[1]
                force_modifier *= masks
            force += force_modifier

        if self.edge_repulsion is not None:
            force_modifier = self.__edge_repulsion(edge_center, H, edge_to_edge_distance) * self.edge_repulsion
            force += force_modifier

        if self.center_gravity is not None:
            masks = [np.zeros((position.shape[0], 1)), np.zeros((position.shape[0], 1))]
            masks[0][:self.nums[0]] = 1
            masks[1][self.nums[0]:] = 1
            for center, gravity, mask in zip(self.centers, self.center_gravity, masks):
                v2c_dist = euclidean_distances(position, center.reshape(1, -1)).reshape(-1, 1)
                force_modifier = self.__center_gravity(position, center, v2c_dist) * gravity * mask
                force += force_modifier

        force *= damping

        force = np.clip(force, Defaults.force_modifier, Defaults.force_a_max)

        position += force * delta

        velocity = force

        return position, velocity, self._stop_condition(velocity, epsilon)

    @staticmethod
    def __node_attraction(position, e_center, vertex_to_edge_dist, x0=0.1, k=1.0):
        x = deepcopy(vertex_to_edge_dist)
        x[vertex_to_edge_dist > 0] -= x0

        f_scale = k * x

        force_direction = e_center[np.newaxis, :, :] - position[:, np.newaxis, :]
        force_direction_length = np.linalg.norm(force_direction, axis=2)
        force_direction = safe_div(force_direction, force_direction_length[:, :, np.newaxis])

        f = f_scale[:, :, np.newaxis] * force_direction
        f = f.sum(axis=1)

        return f

    @staticmethod
    def __node_repulsion(position, vertex_to_vertex_distance, k=1.0):
        distance = vertex_to_vertex_distance.copy()

        r, c = np.diag_indices_from(distance)

        distance[r, c] = np.inf

        force_scale = k / (distance ** 2)
        force_direction = position[:, np.newaxis, :] - position[np.newaxis, :, :]
        force_direction_length = np.linalg.norm(force_direction, axis=2)
        force_direction_length[r, c] = np.inf
        force_direction = safe_div(force_direction, force_direction_length[:, :, np.newaxis])

        force = force_scale[:, :, np.newaxis] * force_direction
        force[r, c] = 0
        force = force.sum(axis=1)

        return force

    @staticmethod
    def __edge_repulsion(edge_center, H, edge_to_edge_dist, k=1.0):
        distance = edge_to_edge_dist.copy()

        r, c = np.diag_indices_from(distance)

        distance[r, c] = np.inf

        force_scale = k / (distance ** 2)
        force_direction = edge_center[:, np.newaxis, :] - edge_center[np.newaxis, :, :]
        force_direction_length = np.linalg.norm(force_direction, axis=2)
        force_direction_length[r, c] = np.inf
        force_direction = safe_div(force_direction, force_direction_length[:, :, np.newaxis])

        force = force_scale[:, :, np.newaxis] * force_direction
        force[r, c] = 0
        force = force.sum(axis=1)

        return np.matmul(H, force)

    @staticmethod
    def __center_gravity(position, center, vertex_to_vertex_distance, k=1):
        force_scale = vertex_to_vertex_distance
        force_direction = center[np.newaxis, np.newaxis, :] - position[:, np.newaxis, :]
        force_direction_length = np.linalg.norm(force_direction, axis=2)
        force_direction = safe_div(force_direction, force_direction_length[:, :, np.newaxis])

        force = force_scale[:, :, np.newaxis] * force_direction
        force = force.sum(axis=1) * k

        return force

    @staticmethod
    def _stop_condition(velocity, epsilon):
        return np.linalg.norm(velocity) < epsilon
