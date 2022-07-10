from typing import Dict, Tuple

import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import MultiAgentObservation, observation_factory
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle
import random
import pprint

class MergeEnv(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """
    GLO_OBS =  {"observation": {"type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "vehicles_count": 5,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],

                    "absolute": True,
                    "flatten": False,
                    "observe_intentions": False,
                    "normalize": False,
                    "see_behind": True                
                },

                }}

    def __init__(self, cfg: dict = None) -> None:
        super().__init__(cfg)
        self.observation_type_global = None

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "ends": [150, 80, 80, 150],
            "collision_reward": -1,
            "arrived_reward": 10,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.2,
            "merging_speed_reward": -0.5,
            "lane_change_reward": -0.05,
            "duration": 100,
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "lateral": True,
                    "longitudinal": True
                }},
            "observation": { "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "vehicles_count": 5,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
  
                    "absolute": False,
                    "flatten": False,
                    "observe_intentions": False,
                    "normalize": False,
                    "see_behind": True
                },
            },
            "controlled_vehicles": 2,
            "initial_vehicle_count": 10
        })
        return cfg

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        super().define_spaces()
        self.observation_type_global = observation_factory(self, self.GLO_OBS["observation"])

    def _reward(self, action: int) -> float:
        rew = []
        for v,a in zip(self.controlled_vehicles, action):
            rew.append(self.agent_reward(a, v))
        ret = []
        for r in rew:
            ret.append(sum(rew))            
        return rew

    def agent_reward(self, action: int, cur_vehicle) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        action_reward = {0: self.config["lane_change_reward"],
                         1: 0,
                         2: self.config["lane_change_reward"],
                         3: 0,
                         4: 0}
        reward = self.config["collision_reward"] * cur_vehicle.crashed \
            + self.has_arrived(cur_vehicle) * self.config["arrived_reward"] \
            + self.config["right_lane_reward"] * cur_vehicle.lane_index[2] / 1 \
            + self.config["high_speed_reward"] * cur_vehicle.speed_index / (cur_vehicle.target_speeds.size - 1)
        # print(self.vehicle.lane_index)
        # Altruistic penalty
        if cur_vehicle.lane_index == ("b", "c", 2):
            reward += self.config["merging_speed_reward"] * \
                        (cur_vehicle.target_speed - cur_vehicle.speed) / cur_vehicle.target_speed
        # print(reward)
        reward += action_reward[action]
        return reward
        return utils.lmap(reward,
                          [self.config["collision_reward"] + self.config["merging_speed_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])

    def has_arrived(self, vehicle):
        if vehicle.lane_index[1] == 'jj':
            # print('arrive jj')
            return 1
        else:
            return 0

    def step2(self,action):
        ret = super().step(action)
        for v in self.road.vehicles:
            if self._agent_is_terminal(v):
                try:
                    self.road.vehicles.remove(v)
                except:
                    pass
        # for v in self.road.vehicles:
        #     if not v.crashed:
        #         if v.position[0]>920:
        #             v.position[0] = 0
        #             lane_index = self.road.network.get_closest_lane_index(v.position)
        #             lane = self.road.network.get_lane(lane_index)
        #             v.lane = lane
        #             v.lane_index = lane_index
        #             v.target_lane_index = lane_index
        return ret
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        _obs = np.array(obs)
        print('feature rel obs: ', np.round(_obs, decimals=0))
        _por = self._get_portal_obs(obs)
        print('feature por obs: ', np.round(_por, decimals=0))
        vec_bvs = _por[:, 1:, 1:3]
        vec_ego = np.expand_dims(_obs[:, 0, 1:3], axis = 1)
        euclidean_to_neighbor = np.linalg.norm((vec_bvs - vec_ego), axis = 2)
        LngRego_bvs = euclidean_to_neighbor* _obs[:, 1:, 5]
        # LatRego_bvs = np.expand_dims(euclidean_to_neighbor, axis = 0) * _obs[:, 1:, 6] # TODO bug
        _obs[:, 1:, 1] = LngRego_bvs
        # _obs[:, 1:, 2] = LatRego_bvs
        # print('vec_ego: ')
        # pprint.pprint(vec_ego)
        # print('euclidean_to_neighbor: ')
        # pprint.pprint(euclidean_to_neighbor)
        # print('LngRego_bvs: ')
        # pprint.pprint(LngRego_bvs)
        obs = _obs # replace obs with portal longitudinal relative obs

        for v in self.road.vehicles:
            if self._agent_is_terminal(v):
                try:
                    self.road.vehicles.remove(v)
                except:
                    pass
        for v in self.road.vehicles:
            if not v.crashed:
                if v.position[0]>920:
                    v.position[0] = 0
                    lane_index = self.road.network.get_closest_lane_index(v.position)
                    lane = self.road.network.get_lane(lane_index)
                    v.lane = lane
                    v.lane_index = lane_index
                    v.target_lane_index = lane_index
        return obs, reward, done, info

    def _get_portal_obs(self, obs: np.ndarray):
    #     """
    #     :return: replace x,y with portal obs
    #     """
        glo_obs = np.array(self.observation_type_global.observe())

        min_x, max_x = 0, 2 * sum(self.config["ends"])
        min_y, max_y = 0, 16.5 # not important in this env, real world size of the grid [[min_x, max_x], [min_y, max_y]]
        world_size = [max_x - min_x, max_y - min_y]
        wall_dists_x = np.minimum(world_size[0] - glo_obs[:, 1:, 1], glo_obs[:, 1:, 1])
        wall_dists_y = np.minimum(world_size[1] - glo_obs[:, 1:, 2], glo_obs[:, 1:, 2])

        glo_obs[:, 1:, 1] = wall_dists_x #  x
        glo_obs[:, 1:, 2] = wall_dists_y #  y

        por_obs = glo_obs
        # wall_angles = np.array([0, np.pi / 2, np.pi, 3 / 2 * np.pi]) - obs[4]
        return por_obs

    def _agent_is_terminal(self, vehicle):
        return vehicle.crashed or self.steps >= self.config["duration"] * self.config["policy_frequency"] #or vehicle.position[0] > 900 or self.has_arrived(vehicle)

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return ([self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles])

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        num_highway = 2
        ends = self.config["ends"]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(num_highway):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))
        # Highway lanes mirro    
        for i in range(num_highway):
            net.add_lane("d", "cc", StraightLane([sum(ends), y[i]], [2*sum(ends)-sum(ends[:3]), y[i]], line_types=line_type[i]))
            net.add_lane("cc", "bb", StraightLane([2*sum(ends)-sum(ends[:3]), y[i]], [2*sum(ends)-sum(ends[:2]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("bb", "aa", StraightLane([2*sum(ends)-sum(ends[:2]), y[i]], [2*sum(ends), y[i]], line_types=line_type[i]))
        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        #  # Merging lane  mirro
        lkkjj = StraightLane([2*sum(ends)-sum(ends[:1]), 6.5 + 4 + 4], [2*sum(ends), 6.5 + 4 + 4], line_types=[c, c], forbidden=False)
        lccbb = StraightLane([2*sum(ends)-sum(ends[:3]), 8], [2*sum(ends)-sum(ends[:2]), 8],
                            line_types=[n, c], forbidden=False)

        lbbkk = SineLane([2*sum(ends)-sum(ends[:2]), 6.5 + 4 + 4 -amplitude], [2*sum(ends)-sum(ends[:1]), 6.5 + 4 + 4 -amplitude],
                       amplitude, 2 * np.pi / (2*ends[1]), -np.pi / 2, line_types=[c, c], forbidden=False)

        net.add_lane("kk", "jj", lkkjj)
        net.add_lane("bb", "kk", lbbkk)
        net.add_lane("cc", "bb", lccbb)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self, n_vehicles: int) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []

        spawn_points_s = [10, 50, 90, 130, 170, 210]
        spawn_points_m = [5, 45, 85, 125, 165, 205]

        """Spawn points for CAV"""
        num_CAV = self.config["controlled_vehicles"]
        num_HDV = n_vehicles - num_CAV
        # spawn point indexes on the straight road
        spawn_point_s_c = np.random.choice(spawn_points_s, num_CAV // 2, replace=False)
        # spawn point indexes on the merging road
        spawn_point_m_c = np.random.choice(spawn_points_m, num_CAV - num_CAV // 2,
                                           replace=False)
        spawn_point_s_c = list(spawn_point_s_c)
        spawn_point_m_c = list(spawn_point_m_c)
        # remove the points to avoid duplicate
        for a in spawn_point_s_c:
            spawn_points_s.remove(a)
        for b in spawn_point_m_c:
            spawn_points_m.remove(b)

        """Spawn points for HDV"""
        # spawn point indexes on the straight road
        spawn_point_s_h = np.random.choice(spawn_points_s, num_HDV // 2, replace=False)
        # spawn point indexes on the merging road
        spawn_point_m_h = np.random.choice(spawn_points_m, num_HDV - num_HDV // 2,
                                           replace=False)
        spawn_point_s_h = list(spawn_point_s_h)
        spawn_point_m_h = list(spawn_point_m_h)

        # initial speed with noise and location noise
        initial_speed = np.random.rand(num_CAV + num_HDV) * 2 + 25  # range from [25, 27]
        loc_noise = np.random.rand(num_CAV + num_HDV) * 3 - 1.5  # range from [-1.5, 1.5]
        initial_speed = list(initial_speed)
        loc_noise = list(loc_noise)

        """spawn the CAV on the straight road first"""
        for _ in range(num_CAV // 2):
            ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("a", "b", 0)).position(
                spawn_point_s_c.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)
        """spawn the rest CAV on the merging road"""
        for _ in range(num_CAV - num_CAV // 2):
            ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("j", "k", 0)).position(
                spawn_point_m_c.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)

        """spawn the HDV on the main road first"""
        for _ in range(num_HDV // 2):
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(
                    spawn_point_s_h.pop(0) + loc_noise.pop(0), 0),
                                    speed=initial_speed.pop(0)))

        """spawn the rest HDV on the merging road"""
        for _ in range(num_HDV - num_HDV // 2):
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(
                    spawn_point_m_h.pop(0) + loc_noise.pop(0), 0),
                                    speed=initial_speed.pop(0)))

    def _make_vehicles2(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        self.controlled_vehicles = []
        for i in range(self.config['controlled_vehicles']):
            ego_vehicle = self.action_type.vehicle_class(road,
                                                        road.network.get_lane(("a", "b", random.randint(0,1))).position(random.randint(0,200), 0),
                                                        speed=30)
            road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(90, 0), speed=29))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), speed=31))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), speed=31.5))

        merging_v = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(110, 0), speed=20)
        merging_v.target_speed = 30
        road.vehicles.append(merging_v)
        # self.vehicle = ego_vehicle


register(
    id='merge-v0',
    entry_point='highway_env.envs:MergeEnv',
)