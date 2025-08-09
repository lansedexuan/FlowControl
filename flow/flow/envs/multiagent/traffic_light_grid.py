"""Multiagent environments for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m traffic light grid.
"""
import math
from random import choice

import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from flow.core import rewards
from flow.envs.traffic_light_grid import TrafficLightGridPOEnv
from flow.envs.multiagent import MultiEnv

ADDITIONAL_ENV_PARAMS = {
    # num of nearby edges the agent can observe {0, ..., num_edges}
    "num_local_edges": 4,
}

ADDITIONAL_ENV_PARAMS_ACCEL = {
    # num of nearby edges the agent can observe {0, ..., num_edges}
    "num_local_edges": 4,
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 2.6,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 7.5,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 50,
}

# Index for retrieving ID when splitting node name, e.g. ":center#"
ID_IDX = 1


class MultiTrafficLightGridPOEnv(TrafficLightGridPOEnv, MultiEnv):
    """Multiagent shared model version of TrafficLightGridPOEnv.

    Required from env_params: See parent class

    States
        See parent class

    Actions
        See parent class

    Rewards
        See parent class

    Termination
        See parent class
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)


class PressLight(MultiTrafficLightGridPOEnv):

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # number of the incoming edges to observe, defaults to 4
        self.num_local_edges = env_params.additional_params.get(
            "num_local_edges", 4)
        # normalizing constants
        edge_length = []
        edge_length.extend([self.k.network.edge_length(edge) for edge in self.k.network.get_edge_list()])
        self.max_length = max(edge_length)

        self.static = self.env_params.additional_params.get("static", False)
        self.observation_info = {}

    @property
    def action_space(self):
        return Discrete(2)

    @property
    def observation_space(self):
        # each incoming/outgoing lane for one intersection
        # road occupancy
        # current phase
        return Box(low=0., high=math.ceil(self.max_length / 7.5), shape=(4 * self.num_local_edges + 1,))

    def get_state(self):
        obs = {}
        phases = {'GrGr': 0, 'yryr': 1, 'rGrG': 2, 'ryry': 3}

        node_to_edges = self.network.node_mapping
        all_incoming_edges = []
        for i in range(len(self.k.traffic_light.get_ids())):
            all_incoming_edges.extend(node_to_edges[i][1])
        veh_num_per_edge = {}  # {name of each edge: road occupancy}
        veh_num_per_edge_split = {}  # {name of each edge segment: #veh}
        veh_num_per_edge_real = {}  # {name of each edge: #veh}
        for each in self.k.network.get_edge_list():
            veh_list = self.k.vehicle.get_ids_by_edge(each)
            w_nor = math.ceil(self.k.network.edge_length(each) / 7.5)
            veh_num_per_edge.update({each: len(veh_list) / w_nor})
            veh_num_per_edge_real.update({each: len(veh_list)})
            if each in all_incoming_edges:
                for i in range(0, 3):
                    veh_num_per_edge_split.update({each + ':' + str(i): 0})
                for each_veh in veh_list:
                    dis_veh = (self.k.network.edge_length(each) -
                               self.k.vehicle.get_position(each_veh)) / \
                              self.k.network.edge_length(each)
                    if dis_veh < 0.34:
                        veh_num_per_edge_split[each + ':0'] += 1
                    elif dis_veh > 0.66:
                        veh_num_per_edge_split[each + ':2'] += 1
                    else:
                        veh_num_per_edge_split[each + ':1'] += 1

        for rl_id in self.k.traffic_light.get_ids():
            rl_id_num = int(rl_id.split("center")[1])
            local_edges = node_to_edges[rl_id_num][1]
            local_edges_out = ['bot'+local_edges[0][3:5]+str(int(local_edges[0].split('_')[1])+1),  # for bot9_9 -> bot9_10
                               'right'+str(int(local_edges[1].split('_')[0].strip('right'))+1)+'_'+local_edges[1].split('_')[1],
                               'top'+local_edges[2][3:5]+str(int(local_edges[2].split('_')[1])-1),
                               'left'+str(int(local_edges[3].split('_')[0].strip('left'))-1)+'_'+local_edges[3].split('_')[1]
                               ]

            veh_num_per_in_0 = [veh_num_per_edge_split[each + ':0'] for each in local_edges]
            veh_num_per_in_1 = [veh_num_per_edge_split[each + ':1'] for each in local_edges]
            veh_num_per_in_2 = [veh_num_per_edge_split[each + ':2'] for each in local_edges]
            veh_num_per_out = [veh_num_per_edge_real[each] for each in local_edges_out]

            veh_num_per_in_nor = [veh_num_per_edge[each] for each in local_edges]
            veh_num_per_out_nor = [veh_num_per_edge[each] for each in local_edges_out]

            # current phase
            phase = self.k.traffic_light.get_state(rl_id)
            phase_index = phases.get(phase) / 3  # normalisation

            observation = np.array(np.concatenate([veh_num_per_in_0, veh_num_per_in_1, veh_num_per_in_2,
                                                   veh_num_per_out, [phase_index]]))
            obs.update({rl_id: observation})
            self.observation_info.update({rl_id: np.array(np.concatenate([veh_num_per_in_nor, veh_num_per_out_nor]))})

        return obs

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        reward = {}
        for rl_id in self.k.traffic_light.get_ids():
            obs = self.observation_info[rl_id]
            w_l_m = []
            for i in range(0, self.num_local_edges):
                w_l_m.append(obs[i] - obs[i + self.num_local_edges])

            reward[rl_id] = -np.abs(np.sum(np.array(w_l_m)))

        return reward

    def _apply_rl_actions(self, rl_actions):
        for rl_id, rl_action in rl_actions.items():
            i = int(rl_id.split("center")[ID_IDX])

            # convert values less than 0.0 to zero and above to 1. 0's
            # indicate that we should not switch the direction
            action = rl_action > 0.0

            non_rl = []
            if self.static:
                non_rl = [i for i in range(0, len(rl_actions.keys()))]

            if self.currently_yellow[i] == 1:  # currently yellow
                self.last_change[i] += self.sim_step
                # Check if our timer has exceeded the yellow phase, meaning it
                # should switch to red
                if self.last_change[i] >= self.min_switch_time:
                    if i not in non_rl:
                        if self.direction[i] == 0:
                            self.k.traffic_light.set_state(
                                node_id='center{}'.format(i), state="GrGr")
                        else:
                            self.k.traffic_light.set_state(
                                node_id='center{}'.format(i), state='rGrG')
                        self.currently_yellow[i] = 0
            else:
                if action:
                    if i not in non_rl:
                        if self.direction[i] == 0:
                            self.k.traffic_light.set_state(
                                node_id='center{}'.format(i), state='yryr')
                        else:
                            self.k.traffic_light.set_state(
                                node_id='center{}'.format(i), state='ryry')
                        self.last_change[i] = 0.0
                        self.direction[i] = not self.direction[i]
                        self.currently_yellow[i] = 1

    def reset(self, **kwargs):
        self.observation_info = {}
        return super().reset()

#cotv
class CoTV(MultiTrafficLightGridPOEnv):

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS_ACCEL.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # number of the incoming edges to observe, defaults to 4
        self.num_local_edges = env_params.additional_params.get("num_local_edges", 4)
        self.target_speed = env_params.additional_params.get("target_velocity", 15)

        # individual veh info {veh_id: [speed, accel, dis, edge]} from real observation space
        self.vehs_edges = {i: {} for i in self.k.network.get_edge_list()}  # vehs list for all edges in the network

        self.observation_info = {}
        self.leader = []

        self.controlled_cav = []
        self.inc_road = []
        for _, edges in self.network.node_mapping:
            self.inc_road.extend(edges)
        # traffic - penetration rate
        self.total_veh = env_params.additional_params.get("total_veh", 240)
        self.cav_num = round(env_params.additional_params.get("cav_penetration_rate", 1) * self.total_veh)
        self.veh_type_set = [1] * self.cav_num + [0] * (self.total_veh - self.cav_num)
        self.veh_type = {}

        # 新增：信号灯切换惩罚相关
        self.last_tl_actions = {tl_id: 0 for tl_id in self.k.traffic_light.get_ids()}
        self.switch_penalty_factor = 0.05  # 可调整参数
        self.max_pressure = 0.0
        self.light_count=0
        self.vehicle_count=0

    @property
    def action_space_tl(self):
        return Discrete(2)

    @property
    def action_space_av(self):
        return Box(low=-abs(self.env_params.additional_params["max_decel"]),
                   high=self.env_params.additional_params["max_accel"], shape=(1,))

    @property
    def observation_space_tl(self):
        """State space that is partially observed.

        Velocities, acceleration, distance to intersections,
        edge number (for the nearest vehicles observed) from each direction,
        local edge information (road occupancy),
        and traffic light state (the passable direction, yellow light flag,
        and the remaining number of seconds for the current yellow light).
        """
        return Box(low=0., high=1, shape=(4 * self.num_local_edges * self.num_observed +
                                          2 * self.num_local_edges + 3,))

    @property
    def observation_space_av(self):
        """See class definition."""
        return Box(low=-5, high=5, shape=(9,))

    def choose_cav_agent(self, sorted_veh_id, edge):
        agent = ""
        # delete CAV agent who has driven across the intersection
        for veh in self.controlled_cav:
            if self.k.vehicle.get_edge(veh) not in self.inc_road:
                self.controlled_cav.remove(veh)
        # keep the same CAV agent
        no_exist = True
        for veh in self.controlled_cav:
            if self.k.vehicle.get_edge(veh) == edge:
                no_exist = False
                agent = veh
                break
        # choose the new CAV agent
        if no_exist:
            if sorted_veh_id:
                for i in range(len(sorted_veh_id)):
                    if sorted_veh_id[i] not in self.veh_type.keys():
                        typee = choice(self.veh_type_set)
                        self.veh_type_set.remove(typee)
                        if typee == 1:
                            self.veh_type.update({sorted_veh_id[i]: typee})
                        else:
                            self.veh_type.update({sorted_veh_id[i]: typee})
                    if self.veh_type[sorted_veh_id[i]] == 1:
                        self.controlled_cav.append(sorted_veh_id[i])
                        agent = sorted_veh_id[i]
                        break
        return agent

    def get_observed_info_veh(self, veh_id, max_speed, max_dist, max_accel):
        max_decel = 15  # emergence stop
        accel_norm = max_accel + max_decel

        speed_veh = self.k.vehicle.get_speed(veh_id) / max_speed
        accel_veh = self.k.vehicle.get_realized_accel(veh_id)
        dis_veh = (self.k.network.edge_length(self.k.vehicle.get_edge(veh_id)) -
                   self.k.vehicle.get_position(veh_id)) / max_dist
        edge_veh = self._convert_edge(self.k.vehicle.get_edge(veh_id)) / (self.k.network.network.num_edges - 1)

        # accel normalization
        if accel_veh < -15:
            accel_veh = -15
        accel_veh = (accel_veh + 15) / accel_norm
        return [speed_veh, accel_veh, dis_veh, edge_veh]

    def get_state(self):
        obs = {}

        """Observations for each TL/CAV agent.

        - For the self.num_observed number of vehicles closest and incoming
        towards traffic light agent, gives the vehicle velocity,  acceleration,
        distance to intersection, edge number.
        - For edges in the network, gives the road occupancy.
        - For the signal phase and timing, gives the traffic light information, including the last
        change time, light direction (i.e. phase), and a currently_yellow flag.
        - For one CAV, speed, acceleration, and headway of leader, distance to intersection, 
        ego speed and its approaching TL
        """

        # Normalization factors
        max_speed = self.k.network.max_speed()
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"], grid_array["long_length"],
                       grid_array["inner_length"])
        max_accel = self.env_params.additional_params["max_accel"]

        # vehicles for incoming and outgoing - info map
        w_max = max_dist / 7.5  # normalization for vehicle number, length + min gap
        veh_num_per_edge = {}  # key: name of each edge in the road network
        for each in self.k.network.get_edge_list():
            all_vehs = self.k.vehicle.get_ids_by_edge(each)
            # remove vehicles already left the edge after one step, not restore at this step
            pre_observed_vehs = list(self.vehs_edges[each].keys())
            for each_veh in pre_observed_vehs:
                if each_veh not in all_vehs:
                    del self.vehs_edges[each][each_veh]
            # update at this step
            for veh in all_vehs:  # get the info for updated vehicles
                self.vehs_edges[each].update({veh: self.get_observed_info_veh(veh, max_speed, max_dist, max_accel)})
            veh_num_per_edge.update({each: len(self.vehs_edges[each].keys()) / w_max})

        # Observed vehicle information
        speeds = []
        accels = []
        dist_to_intersec = []
        edge_number = []
        all_observed_ids = []  # [[],[]] observed list for each edge
        for _, edges in self.network.node_mapping:  # each intersection
            local_speeds = []
            local_accels = []
            local_dists_to_intersec = []
            local_edge_numbers = []
            for edge in edges:
                # sort to select the closest vehicle
                veh_id_sort = {}
                for veh in self.vehs_edges[edge].keys():
                    veh_id_sort.update({int(veh.split('.')[1]): veh})
                num_observed = min(self.num_observed, len(self.vehs_edges[edge]))
                sorted_veh_id = []
                for i in range(0, len(veh_id_sort)):
                    sorted_veh_id.append(veh_id_sort[sorted(veh_id_sort.keys())[i]])
                observed_ids = [sorted_veh_id[i] for i in range(0, num_observed)]

                self.choose_cav_agent(sorted_veh_id, edge)

                all_observed_ids.extend(observed_ids)
                local_speeds.extend([self.vehs_edges[edge][veh_id][0] for veh_id in observed_ids])
                local_accels.extend([self.vehs_edges[edge][veh_id][1] for veh_id in observed_ids])
                local_dists_to_intersec.extend([self.vehs_edges[edge][veh_id][2] for veh_id in observed_ids])
                local_edge_numbers.extend([self.vehs_edges[edge][veh_id][3] for veh_id in observed_ids])

                if len(observed_ids) < self.num_observed:
                    diff = self.num_observed - len(observed_ids)
                    local_speeds.extend([1] * diff)
                    local_accels.extend([0] * diff)
                    local_dists_to_intersec.extend([1] * diff)
                    local_edge_numbers.extend([0] * diff)

            speeds.append(local_speeds)
            accels.append(local_accels)
            dist_to_intersec.append(local_dists_to_intersec)
            edge_number.append(local_edge_numbers)
        self.observed_ids = all_observed_ids
        # Traffic light information
        last_change = self.last_change.flatten()  # to control yellow light duration
        direction = self.direction.flatten()
        currently_yellow = self.currently_yellow.flatten()
        # This is a catch-all for when the relative_node method returns a -1
        # (when there is no node in the direction sought). We add a last
        # item to the lists here, which will serve as a default value.
        last_change = np.divide(np.append(last_change, [0]), 3)  # normalization, default is 5 sec yellow light
        direction = np.append(direction, [0])  # default is NS direction (the first in fixed TL cycle)
        currently_yellow = np.append(currently_yellow, [1])  # if there is no traffic light
        # the incoming TL for each CAV
        incoming_tl = {each: "" for each in self.controlled_cav}
        # agent_TL
        node_to_edges = self.network.node_mapping
        for rl_id in self.k.traffic_light.get_ids():
            rl_id_num = int(rl_id.split("center")[ID_IDX])
            local_edges = node_to_edges[rl_id_num][1]
            local_edges_out = ['bot' + local_edges[0][3:5] + str(int(local_edges[0].split('_')[1]) + 1),
                               'right' + str(int(local_edges[1].split('_')[0].strip('right')) + 1) + '_' +
                               local_edges[1].split('_')[1],
                               'top' + local_edges[2][3:5] + str(int(local_edges[2].split('_')[1]) - 1),
                               'left' + str(int(local_edges[3].split('_')[0].strip('left')) - 1) + '_' +
                               local_edges[3].split('_')[1]
                               ]
            veh_num_per_in = [veh_num_per_edge[each] for each in local_edges]
            veh_num_per_out = [veh_num_per_edge[each] for each in local_edges_out]

            for av_id in incoming_tl.keys():
                if self.k.vehicle.get_edge(av_id) in local_edges:
                    incoming_tl[av_id] = rl_id_num  # get the id of the approaching TL

            con = [round(i, 8) for i in np.concatenate(
                [speeds[rl_id_num], accels[rl_id_num], dist_to_intersec[rl_id_num],
                 edge_number[rl_id_num],
                 veh_num_per_in, veh_num_per_out,
                 last_change[[rl_id_num]],
                 direction[[rl_id_num]], currently_yellow[[rl_id_num]]])]

            observation = np.array(con)
            obs.update({rl_id: observation})

        # agent_CAV information
        for rl_id in self.controlled_cav:
            this_pos = self.k.network.edge_length(self.k.vehicle.get_edge(rl_id)) - self.k.vehicle.get_position(
                rl_id)
            this_speed = self.k.vehicle.get_speed(rl_id)
            this_accel = self.k.vehicle.get_realized_accel(rl_id)
            this_accel = (this_accel, -15)[abs(this_accel) >= 15]

            lead_id = self.k.vehicle.get_leader(rl_id)
            self.leader = []

            if incoming_tl[rl_id] != "":
                incoming_tl_id = int(incoming_tl[rl_id])
            else:
                incoming_tl_id = -1  # set default value

            if lead_id in ["", None] or self.k.vehicle.get_speed(lead_id) == -1001:
                # in case leader is not visible
                lead_speed = max_speed + this_speed
                lead_head = max_dist
                lead_accel = 15
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_headway(rl_id)
                lead_head = (lead_head, 1000)[lead_head > 1000]
                lead_accel = self.k.vehicle.get_realized_accel(lead_id)
                lead_accel = (lead_accel, -15)[abs(lead_accel) >= 15]

            obs.update({rl_id: np.array([
                this_pos / max_dist,
                this_speed / max_speed,
                this_accel / max_accel,
                (lead_speed - this_speed) / max_speed,
                lead_head / max_dist,
                lead_accel / max_accel,
                last_change[incoming_tl_id],
                direction[incoming_tl_id],
                currently_yellow[incoming_tl_id]])})

        self.observation_info = obs
        return obs

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        reward = {}
        all_pressures = []
        for rl_id in self.k.traffic_light.get_ids():
            obs = self.observation_info[rl_id]
            # 1. pressure
            traffic_start = 4 * self.num_local_edges * self.num_observed
            inc_traffic = np.sum(obs[traffic_start: traffic_start + self.num_local_edges])
            out_traffic = np.sum(obs[traffic_start + self.num_local_edges:
                                     traffic_start + self.num_local_edges * 2])
            pressure = inc_traffic - out_traffic

            all_pressures.append(pressure)
            # 计算平均压力作为动态阈值基准
            avg_pressure = np.mean(all_pressures)
            # 归一化基准（最大压力取平均压力的2倍，可根据实际情况调整）
            self.max_pressure = max(self.max_pressure, avg_pressure * 2)

            # 计算当前路口压力并归一化
            pressure = all_pressures[int(rl_id.split("center")[ID_IDX])]
            norm_pressure = min(max(pressure / self.max_pressure, 0.0), 1.0)

            # 使用动态阈值（基于全局平均压力）
            threshold = 0.6 + 0.2 * (avg_pressure / self.max_pressure)  # 阈值范围[0.6,0.8]

            current_action = rl_actions.get(rl_id, 0) > 0.0
            # 根据归一化压力和动态阈值调整惩罚
            if norm_pressure > threshold:
                switch_penalty = 0  # 压力超过阈值时无惩罚
            else:
                # 压力越小，惩罚系数越大（但惩罚上限随全局压力增加而降低）
                penalty_factor = self.switch_penalty_factor * (1 - norm_pressure) * ( 2 - avg_pressure / self.max_pressure )
                switch_penalty = penalty_factor  if current_action != self.last_tl_actions[rl_id] else 0

            # 新增：信号灯切换惩罚
            #current_action = rl_actions.get(rl_id, 0) > 0.0
            #switch_penalty = self.switch_penalty_factor if current_action != self.last_tl_actions[rl_id] else 0

            reward[rl_id] = -pressure #-switch_penalty  # 惩罚: 压力 信号切换

            '''
            self.light_count+=1
            if self.light_count==3000:
                # ===== 新增：打印交通灯奖励分量 =====
                print(f"1 [TL {rl_id}] Pressure: {pressure:.4f}, Switch Penalty: {switch_penalty:.4f}")
                print(f"1 [TL {rl_id}] Reward: {reward[rl_id]:.4f}")
                print("-" * 40)
                self.light_count=0
            '''

        for rl_id in self.controlled_cav:
            edge = self.k.vehicle.get_edge(rl_id)# 获取当前边缘上的所有车辆ID
            veh_ids = self.k.vehicle.get_ids_by_edge(edge)

            # 计算延迟奖励
            delay_reward = rewards.min_delay_edge(self, veh_ids, self.target_speed) - 1

            # 计算加速度惩罚
            accel_penalty = rewards.stable_acceleration_positive_edge(self, veh_ids)

            # 新增：碰撞风险惩罚
            collision_penalty = 0.0
            decel_risk = 0.0
            lead_id = self.k.vehicle.get_leader(rl_id)
            if lead_id and lead_id != rl_id:
                headway = self.k.vehicle.get_headway(rl_id)
                ego_speed = self.k.vehicle.get_speed(rl_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                # 新增：前车加速度（若为负，说明前车在减速）
                lead_decel = self.k.vehicle.get_accel(lead_id)  # 从之前的代码中获取lead_accel

                # 计算理论安全距离（反应距离 + 制动距离）
                safe_distance = max(1.0, ego_speed * 1.0 + (ego_speed ** 2) / (2 * 3.0))
                # 计算相对速度（负值表示接近）
                rel_speed = lead_speed - ego_speed

                if lead_decel is None:
                    lead_decel = 0
                if lead_decel < 0:  # 前车在减速
                    # 额外增加风险评估，因为前车减速可能导致本车必须更急地减速
                    # 获取环境参数中的最大减速度（假设self.env_params.additional_params已包含max_decel）
                    max_decel = self.env_params.additional_params.get("max_decel", 7.5)  # 默认值7.5，与ADDITIONAL_ENV_PARAMS_ACCEL保持一致

                    # 归一化前车减速度（转为正数便于计算）
                    normalized_decel = min(abs(lead_decel) / max_decel, 1.0)  # 限制在[0,1]

                    # 计算风险值（使用指数函数平滑，避免线性增长）
                    decel_risk = 1.0 - np.exp(-2 * normalized_decel)  # [0,1]，归一化减速度越大，风险越高

                # 当实际距离 < 安全距离 或 相对速度为负（正在接近）时，计算风险
                if headway < safe_distance or rel_speed < 0:
                    # 计算TTC（碰撞时间），避免除以零
                    ttc = headway / max(0.1, abs(rel_speed)) if rel_speed <= 0 else float('inf')

                    # 归一化风险分数：距离越近、TTC越小，风险越高
                    distance_risk = 1 - min(headway / safe_distance, 1.0)  # [0,1]
                    ttc_risk = np.exp(-ttc)  # [0,1]，TTC越大风险越低

                    # 综合风险：同时考虑距离和TTC
                    combined_risk = max(distance_risk, ttc_risk, decel_risk)

                    # 平滑惩罚：使用指数函数放大高风险区域
                    collision_penalty = 1.0 - np.exp(-2 * combined_risk)  # [0,1]

            reward[rl_id] = delay_reward - accel_penalty #- collision_penalty

            '''
            self.vehicle_count+=1
            if self.vehicle_count==3000:
                # ===== 新增：打印CAV奖励分量 =====
                print(f"2 [CAV {rl_id}] Delay Reward: {delay_reward:.4f}, Accel Penalty: {accel_penalty:.4f}")
                print(f"2 [CAV {rl_id}] Collision Penalty: : {collision_penalty:.4f}")
                print(f"2 [CAV {rl_id}] Final Reward: {reward[rl_id]:.4f}")
                print("-" * 40)
                self.vehicle_count=0
            '''

        return reward


    def _apply_rl_actions(self, rl_actions):
        for rl_id, rl_action in rl_actions.items():
            if "center" in rl_id:  # TL
                i = int(rl_id.split("center")[ID_IDX])
                action = rl_action > 0.0

                if self.currently_yellow[i] == 1:  # currently yellow
                    self.last_change[i] += self.sim_step
                    # Check if our timer has exceeded the yellow phase, meaning it
                    # should switch to red]))
                    if round(float(self.last_change[i]), 8) >= self.min_switch_time:
                        if self.direction[i] == 0:
                            self.k.traffic_light.set_state(
                                node_id='center{}'.format(i), state="GrGr")
                        else:
                            self.k.traffic_light.set_state(
                                node_id='center{}'.format(i), state='rGrG')
                        self.currently_yellow[i] = 0
                else:
                    if action.any():
                        if self.direction[i] == 0:
                            self.k.traffic_light.set_state(
                                node_id='center{}'.format(i), state='yryr')
                        else:
                            self.k.traffic_light.set_state(
                                node_id='center{}'.format(i), state='ryry')
                        self.last_change[i] = 0.0
                        self.direction[i] = not self.direction[i]  # direction-GrGr:0; yryr :1
                        self.currently_yellow[i] = 1
            else:
                self.k.vehicle.apply_acceleration(rl_id, rl_actions[rl_id])

    def reset(self, **kwargs):
        """
        In addition, a few variables that are specific to this class are
        emptied before they are used by the new rollout.
        """
        self.leader = []
        self.observation_info = {}
        self.controlled_cav = []
        self.vehs_edges = {i: {} for i in self.k.network.get_edge_list()}
        self.veh_type_set = [1] * self.cav_num + [0] * (self.total_veh - self.cav_num)
        self.veh_type = {}
        return super().reset()

    def additional_command(self):
        # specify observed vehicles
        for veh_id in self.k.vehicle.get_ids():
            self.k.vehicle.set_color(veh_id=veh_id, color=(255, 255, 255))
        for veh_id in self.controlled_cav:
            self.k.vehicle.set_color(veh_id=veh_id, color=(255, 0, 0))


class CoTVAll(CoTV):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS_ACCEL.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # number of the incoming edges to observe, defaults to 4
        self.num_local_edges = env_params.additional_params.get("num_local_edges", 4)
        self.target_speed = env_params.additional_params.get("target_velocity", 15)

        self.observation_info = {}
        self.leader = []

    def get_state(self):
        obs = {}

        # Normalization factors
        max_speed = self.k.network.max_speed()
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"], grid_array["long_length"],
                       grid_array["inner_length"])
        max_accel = self.env_params.additional_params["max_accel"]
        max_decel = 15  # emergence stop
        accel_norm = max_accel + max_decel

        # Observed vehicle information
        speeds = []
        accels = []
        dist_to_intersec = []
        edge_number = []
        all_observed_ids = []
        for _, edges in self.network.node_mapping:  # each intersection
            local_speeds = []
            local_accels = []
            local_dists_to_intersec = []
            local_edge_numbers = []
            for edge in edges:
                observed_ids = \
                    self.get_closest_to_intersection(edge, self.num_observed)
                all_observed_ids.extend(observed_ids)

                local_speeds.extend(
                    [self.k.vehicle.get_speed(veh_id) / max_speed for veh_id in
                     observed_ids])
                local_accels.extend(
                    [self.k.vehicle.get_realized_accel(veh_id) for veh_id in observed_ids])
                local_dists_to_intersec.extend([(self.k.network.edge_length(
                    self.k.vehicle.get_edge(
                        veh_id)) - self.k.vehicle.get_position(
                    veh_id)) / max_dist for veh_id in observed_ids])
                local_edge_numbers.extend([self._convert_edge(
                    self.k.vehicle.get_edge(veh_id)) / (
                                                   self.k.network.network.num_edges - 1) for veh_id in
                                           observed_ids])

                if len(observed_ids) < self.num_observed:
                    diff = self.num_observed - len(observed_ids)
                    local_speeds.extend([1] * diff)
                    local_accels.extend([0] * diff)
                    local_dists_to_intersec.extend([1] * diff)
                    local_edge_numbers.extend([0] * diff)

            # accel normalization
            for i in range(0, len(local_accels)):
                if local_accels[i] < -15:
                    local_accels[i] = -15
                local_accels[i] = (local_accels[i] + 15) / accel_norm

            speeds.append(local_speeds)
            accels.append(local_accels)
            dist_to_intersec.append(local_dists_to_intersec)
            edge_number.append(local_edge_numbers)
        self.observed_ids = all_observed_ids

        # Traffic light information
        last_change = self.last_change.flatten()  # to control yellow light duration
        direction = self.direction.flatten()
        currently_yellow = self.currently_yellow.flatten()
        # This is a catch-all for when the relative_node method returns a -1
        # (when there is no node in the direction sought). We add a last
        # item to the lists here, which will serve as a default value.
        last_change = np.divide(np.append(last_change, [0]), 3)  # normalization, default is 5 sec yellow light
        direction = np.append(direction, [0])  # default is NS direction (the first in fixed TL cycle)
        currently_yellow = np.append(currently_yellow, [1])  # if there is no traffic light

        # the incoming TL for each CAV
        incoming_tl = {each: "" for each in self.k.vehicle.get_rl_ids()}

        # number of vehicles
        edges = self.k.network.get_edge_list()
        w_max = max_dist / 7.5  # normalization for vehicle number
        veh_num_per_edge = {each: len(self.k.vehicle.get_ids_by_edge(each)) / w_max for each in edges}
        # agent_TL
        node_to_edges = self.network.node_mapping
        for rl_id in self.k.traffic_light.get_ids():
            rl_id_num = int(rl_id.split("center")[ID_IDX])
            local_edges = node_to_edges[rl_id_num][1]
            local_edges_out = ['bot' + local_edges[0][3:5] + str(int(local_edges[0][5]) + 1),
                               'right' + str(int(local_edges[1][5]) + 1) + local_edges[1][6:],
                               'top' + local_edges[2][3:5] + str(int(local_edges[2][5]) - 1),
                               'left' + str(int(local_edges[3][4]) - 1) + local_edges[3][5:]]
            veh_num_per_in = [veh_num_per_edge[each] for each in local_edges]
            veh_num_per_out = [veh_num_per_edge[each] for each in local_edges_out]

            for av_id in incoming_tl.keys():
                if self.k.vehicle.get_edge(av_id) in local_edges:
                    incoming_tl[av_id] = rl_id_num  # get the id of the approaching TL

            con = [round(i, 8) for i in np.concatenate(
                [speeds[rl_id_num], accels[rl_id_num], dist_to_intersec[rl_id_num],
                 edge_number[rl_id_num],
                 veh_num_per_in, veh_num_per_out,
                 last_change[[rl_id_num]],
                 direction[[rl_id_num]], currently_yellow[[rl_id_num]]])]

            observation = np.array(con)
            obs.update({rl_id: observation})
        self.observation_info = obs

        # agent_CAV information
        for rl_id in self.k.vehicle.get_rl_ids():
            this_pos = self.k.network.edge_length(self.k.vehicle.get_edge(rl_id)) - self.k.vehicle.get_position(
                rl_id)
            this_speed = self.k.vehicle.get_speed(rl_id)
            this_accel = self.k.vehicle.get_realized_accel(rl_id)
            this_accel = (this_accel, -15)[abs(this_accel) >= 15]
            lead_id = self.k.vehicle.get_leader(rl_id)

            if incoming_tl[rl_id] != "":
                incoming_tl_id = int(incoming_tl[rl_id])
            else:
                incoming_tl_id = -1  # set default value

            if lead_id in ["", None] or self.k.vehicle.get_speed(lead_id) == -1001:
                # in case leader is not visible
                lead_speed = max_speed + this_speed
                lead_head = max_dist
                lead_accel = 15
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_headway(rl_id)
                lead_accel = self.k.vehicle.get_realized_accel(lead_id)
                lead_accel = (lead_accel, -15)[abs(lead_accel) >= 15]

            obs.update({rl_id: np.array([
                this_pos / max_dist,
                this_speed / max_speed,
                this_accel / max_accel,
                (lead_speed - this_speed) / max_speed,
                lead_head / max_dist,
                lead_accel / max_accel,
                last_change[incoming_tl_id],
                direction[incoming_tl_id],
                currently_yellow[incoming_tl_id]])})
        return obs

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        reward = {}
        for rl_id in self.k.traffic_light.get_ids():
            obs = self.observation_info[rl_id]
            # pressure
            traffic_start = 4 * self.num_local_edges * self.num_observed
            inc_traffic = np.sum(obs[traffic_start: traffic_start + self.num_local_edges])
            out_traffic = np.sum(obs[traffic_start + self.num_local_edges:
                                     traffic_start + self.num_local_edges * 2])
            reward[rl_id] = -(inc_traffic - out_traffic)

        for rl_id in self.k.vehicle.get_rl_ids():
            edge = self.k.vehicle.get_edge(rl_id)
            veh_ids = self.k.vehicle.get_ids_by_edge(edge)
            reward[rl_id] = rewards.min_delay_edge(self, veh_ids, self.target_speed) - 1 \
                - rewards.stable_acceleration_positive_edge(self, veh_ids)

        return reward

    def _apply_rl_actions(self, rl_actions):
        """
        Issues action for each agent(TL / CAV).
        """
        for rl_id, rl_action in rl_actions.items():
            if "center" in rl_id:  # TL
                i = int(rl_id.split("center")[ID_IDX])
                # convert values less than 0.0 to zero and above to 1. 0's
                # indicate that we should not switch the direction
                action = rl_action > 0.0

                if self.currently_yellow[i] == 1:  # currently yellow
                    self.last_change[i] += self.sim_step
                    # Check if our timer has exceeded the yellow phase, meaning it
                    # should switch to red
                    if round(float(self.last_change[i]), 8) >= self.min_switch_time:
                        if self.direction[i] == 0:
                            self.k.traffic_light.set_state(
                                node_id='center{}'.format(i), state="GrGr")
                        else:
                            self.k.traffic_light.set_state(
                                node_id='center{}'.format(i), state='rGrG')
                        self.currently_yellow[i] = 0
                else:
                    if action:
                        if self.direction[i] == 0:
                            self.k.traffic_light.set_state(
                                node_id='center{}'.format(i), state='yryr')
                        else:
                            self.k.traffic_light.set_state(
                                node_id='center{}'.format(i), state='ryry')
                        self.last_change[i] = 0.0
                        self.direction[i] = not self.direction[i]  # direction-GrGr:0; yryr :1
                        self.currently_yellow[i] = 1
            else:
                self.k.vehicle.apply_acceleration(rl_id, rl_actions[rl_id])

    def reset(self, **kwargs):
        self.leader = []
        self.observation_info = {}
        return super().reset()

    def additional_command(self):
        # specify observed vehicles
        for veh_id in self.k.vehicle.get_ids():
            self.k.vehicle.set_color(veh_id=veh_id, color=(255, 0, 0))


class CoTVNOCoord(MultiTrafficLightGridPOEnv):

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS_ACCEL.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # number of the incoming edges to observe, defaults to 4
        self.num_local_edges = env_params.additional_params.get("num_local_edges", 4)
        self.target_speed = env_params.additional_params.get("target_velocity", 15)

        # individual veh info {veh_id: [speed, accel, dis, edge]} from real observation space
        self.vehs_edges = {i: {} for i in self.k.network.get_edge_list()}  # vehs list for all edges in the network

        self.observation_info = {}
        self.leader = []

        self.controlled_cav = []
        self.inc_road = []
        for _, edges in self.network.node_mapping:
            self.inc_road.extend(edges)
        # traffic - penetration rate
        self.total_veh = env_params.additional_params.get("total_veh", 240)
        self.cav_num = round(env_params.additional_params.get("cav_penetration_rate", 1) * self.total_veh)
        self.veh_type_set = [1] * self.cav_num + [0] * (self.total_veh - self.cav_num)
        self.veh_type = {}

    @property
    def action_space_tl(self):
        return Discrete(2)

    @property
    def action_space_av(self):
        return Box(low=-abs(self.env_params.additional_params["max_decel"]),
                   high=self.env_params.additional_params["max_accel"], shape=(1,))

    @property
    def observation_space_tl(self):
        """State space that is partially observed.

        Velocities, acceleration, distance to intersections,
        edge number (for the nearest vehicles observed) from each direction,
        local edge information (road occupancy),
        and traffic light state (the passable direction, yellow light flag,
        and the remaining number of seconds for the current yellow light).
        """
        return Box(low=0., high=1, shape=(2 * self.num_local_edges + 3,))

    @property
    def observation_space_av(self):
        """See class definition."""
        return Box(low=-5, high=5, shape=(6,))

    def choose_cav_agent(self, sorted_veh_id, edge):
        agent = ""
        # delete CAV agent who has driven across the intersection
        for veh in self.controlled_cav:
            if self.k.vehicle.get_edge(veh) not in self.inc_road:
                self.controlled_cav.remove(veh)
        # keep the same CAV agent
        no_exist = True
        for veh in self.controlled_cav:
            if self.k.vehicle.get_edge(veh) == edge:
                no_exist = False
                agent = veh
                break
        # choose the new CAV agent
        if no_exist:
            if sorted_veh_id:
                for i in range(len(sorted_veh_id)):
                    if sorted_veh_id[i] not in self.veh_type.keys():
                        typee = choice(self.veh_type_set)
                        self.veh_type_set.remove(typee)
                        if typee == 1:
                            self.veh_type.update({sorted_veh_id[i]: typee})
                        else:
                            self.veh_type.update({sorted_veh_id[i]: typee})
                    if self.veh_type[sorted_veh_id[i]] == 1:
                        self.controlled_cav.append(sorted_veh_id[i])
                        agent = sorted_veh_id[i]
                        break
        return agent

    def get_observed_info_veh(self, veh_id, max_dist):
        dis_veh = (self.k.network.edge_length(self.k.vehicle.get_edge(veh_id)) -
                   self.k.vehicle.get_position(veh_id)) / max_dist

        return dis_veh

    def get_state(self):
        obs = {}

        """Observations for each TL/CAV agent.

        - For edges in the network, gives the road occupancy.
        - For the signal phase and timing, gives the traffic light information, including the last
        change time, light direction (i.e. phase), and a currently_yellow flag.
        - For one CAV, speed, acceleration, and headway of leader, distance to intersection, 
        ego speed
        """

        # Normalization factors
        max_speed = self.k.network.max_speed()
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"], grid_array["long_length"],
                       grid_array["inner_length"])
        max_accel = self.env_params.additional_params["max_accel"]

        # vehicles for incoming and outgoing - info map
        w_max = max_dist / 7.5  # normalization for vehicle number, length + min gap
        veh_num_per_edge = {}  # key: name of each edge in the road network
        for each in self.k.network.get_edge_list():
            all_vehs = self.k.vehicle.get_ids_by_edge(each)
            # remove vehicles already left the edge after one step, not restore at this step
            pre_observed_vehs = list(self.vehs_edges[each].keys())
            for each_veh in pre_observed_vehs:
                if each_veh not in all_vehs:
                    del self.vehs_edges[each][each_veh]
            # update at this step
            for veh in all_vehs:  # get the info for updated vehicles
                self.vehs_edges[each].update({veh: self.get_observed_info_veh(veh, max_dist)})
            veh_num_per_edge.update({each: len(self.vehs_edges[each].keys()) / w_max})

        # Observed vehicle
        all_observed_ids = []  # [[],[]] observed list for each edge
        for _, edges in self.network.node_mapping:  # each intersection
            for edge in edges:
                # sort to select the closest vehicle
                veh_id_sort = {}
                for veh in self.vehs_edges[edge].keys():
                    veh_id_sort.update({int(veh.split('.')[1]): veh})
                num_observed = min(self.num_observed, len(self.vehs_edges[edge]))
                sorted_veh_id = []
                for i in range(0, len(veh_id_sort)):
                    sorted_veh_id.append(veh_id_sort[sorted(veh_id_sort.keys())[i]])
                observed_ids = [sorted_veh_id[i] for i in range(0, num_observed)]

                self.choose_cav_agent(sorted_veh_id, edge)

                all_observed_ids.extend(observed_ids)

        self.observed_ids = all_observed_ids
        # Traffic light information
        last_change = self.last_change.flatten()  # to control yellow light duration
        direction = self.direction.flatten()
        currently_yellow = self.currently_yellow.flatten()
        # This is a catch-all for when the relative_node method returns a -1
        # (when there is no node in the direction sought). We add a last
        # item to the lists here, which will serve as a default value.
        last_change = np.divide(np.append(last_change, [0]), 3)  # normalization, default is 5 sec yellow light
        direction = np.append(direction, [0])  # default is NS direction (the first in fixed TL cycle)
        currently_yellow = np.append(currently_yellow, [1])  # if there is no traffic light

        # agent_TL
        node_to_edges = self.network.node_mapping
        for rl_id in self.k.traffic_light.get_ids():
            rl_id_num = int(rl_id.split("center")[ID_IDX])
            local_edges = node_to_edges[rl_id_num][1]
            local_edges_out = ['bot' + local_edges[0][3:5] + str(int(local_edges[0][5]) + 1),
                               'right' + str(int(local_edges[1][5]) + 1) + local_edges[1][6:],
                               'top' + local_edges[2][3:5] + str(int(local_edges[2][5]) - 1),
                               'left' + str(int(local_edges[3][4]) - 1) + local_edges[3][5:]]
            veh_num_per_in = [veh_num_per_edge[each] for each in local_edges]
            veh_num_per_out = [veh_num_per_edge[each] for each in local_edges_out]

            con = [round(i, 8) for i in np.concatenate(
                [veh_num_per_in, veh_num_per_out,
                 last_change[[rl_id_num]],
                 direction[[rl_id_num]], currently_yellow[[rl_id_num]]])]

            observation = np.array(con)
            obs.update({rl_id: observation})

        # agent_CAV information
        for rl_id in self.controlled_cav:
            this_pos = self.k.network.edge_length(self.k.vehicle.get_edge(rl_id)) - self.k.vehicle.get_position(
                rl_id)
            this_speed = self.k.vehicle.get_speed(rl_id)
            this_accel = self.k.vehicle.get_realized_accel(rl_id)
            this_accel = (this_accel, -15)[abs(this_accel) >= 15]

            lead_id = self.k.vehicle.get_leader(rl_id)
            self.leader = []

            if lead_id in ["", None] or self.k.vehicle.get_speed(lead_id) == -1001:
                # in case leader is not visible
                lead_speed = max_speed + this_speed
                lead_head = max_dist
                lead_accel = 15
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_headway(rl_id)
                lead_head = (lead_head, 1000)[lead_head > 1000]
                lead_accel = self.k.vehicle.get_realized_accel(lead_id)
                lead_accel = (lead_accel, -15)[abs(lead_accel) >= 15]

            obs.update({rl_id: np.array([
                this_pos / max_dist,
                this_speed / max_speed,
                this_accel / max_accel,
                (lead_speed - this_speed) / max_speed,
                lead_head / max_dist,
                lead_accel / max_accel])})

        self.observation_info = obs
        return obs

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        reward = {}
        for rl_id in self.k.traffic_light.get_ids():
            obs = self.observation_info[rl_id]
            # pressure
            inc_traffic = np.sum(obs[:self.num_local_edges])
            out_traffic = np.sum(obs[self.num_local_edges:self.num_local_edges * 2])
            reward[rl_id] = -(inc_traffic - out_traffic)

        for rl_id in self.controlled_cav:
            edge = self.k.vehicle.get_edge(rl_id)
            veh_ids = self.k.vehicle.get_ids_by_edge(edge)
            reward[rl_id] = rewards.min_delay_edge(self, veh_ids, self.target_speed) - 1 \
                - rewards.stable_acceleration_positive_edge(self, veh_ids)
        return reward

    def _apply_rl_actions(self, rl_actions):
        for rl_id, rl_action in rl_actions.items():
            if "center" in rl_id:  # TL
                i = int(rl_id.split("center")[ID_IDX])
                action = rl_action > 0.0

                if self.currently_yellow[i] == 1:  # currently yellow
                    self.last_change[i] += self.sim_step
                    # Check if our timer has exceeded the yellow phase, meaning it
                    # should switch to red]))
                    if round(float(self.last_change[i]), 8) >= self.min_switch_time:
                        if self.direction[i] == 0:
                            self.k.traffic_light.set_state(
                                node_id='center{}'.format(i), state="GrGr")
                        else:
                            self.k.traffic_light.set_state(
                                node_id='center{}'.format(i), state='rGrG')
                        self.currently_yellow[i] = 0
                else:
                    if action.any():
                        if self.direction[i] == 0:
                            self.k.traffic_light.set_state(
                                node_id='center{}'.format(i), state='yryr')
                        else:
                            self.k.traffic_light.set_state(
                                node_id='center{}'.format(i), state='ryry')
                        self.last_change[i] = 0.0
                        self.direction[i] = not self.direction[i]  # direction-GrGr:0; yryr :1
                        self.currently_yellow[i] = 1
            else:
                self.k.vehicle.apply_acceleration(rl_id, rl_actions[rl_id])

    def reset(self, **kwargs):
        """
        In addition, a few variables that are specific to this class are
        emptied before they are used by the new rollout.
        """
        self.leader = []
        self.observation_info = {}
        self.controlled_cav = []
        self.vehs_edges = {i: {} for i in self.k.network.get_edge_list()}
        self.veh_type_set = [1] * self.cav_num + [0] * (self.total_veh - self.cav_num)
        self.veh_type = {}
        return super().reset()

    def additional_command(self):
        # specify observed vehicles
        for veh_id in self.k.vehicle.get_ids():
            self.k.vehicle.set_color(veh_id=veh_id, color=(255, 255, 255))
        for veh_id in self.controlled_cav:
            self.k.vehicle.set_color(veh_id=veh_id, color=(255, 0, 0))
