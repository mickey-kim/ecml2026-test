from collections import deque
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.step_utils.states import TrainState
from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax


class EnhancedTreeObsBuilder(ObservationBuilder):
    """
    Cooperative Platoon- & Congestion-Aware TreeObsBuilder (136 dims + 5 mask):
    - SELF_DIM = 52: adds opp_count, behind_count, platoon_advantage, opp_slack_diff, team_done_ratio.
    - NEIGHBOR_K = 12, NEIGHBOR_DIM = 7: adds other_rem_dist for multi-agent priority reasoning.
    - Pass-through corridor scan: counts all oncoming & same-direction trains rather than stopping at first train.
    """

    SELF_DIM = 52
    NEIGHBOR_K = 12
    NEIGHBOR_DIM = 7
    OBSERVATION_DIM = SELF_DIM + NEIGHBOR_K * NEIGHBOR_DIM  # 136
    ACTION_MASK_SIZE = 5

    def __init__(self, max_depth=3, with_action_mask=True):
        self.max_depth = max_depth
        self.with_action_mask = with_action_mask
        self.waypoint_index = {}
        self._waypoint_distance_maps = {}
        self._switches_built = False
        self.switches = {}
        self.switches_neighbours = {}
        self._spawn_schedule = {}

    def build_data(self):
        if self.env is not None:
            self.env.dev_obs_dict = {}
        self.switches = {}
        self.switches_neighbours = {}
        self._switches_built = False
        if self.env is not None and self.env.rail is not None:
            self.find_all_cell_where_agent_can_choose()
            self._switches_built = True

    def find_all_cell_where_agent_can_choose(self):
        switches = {}
        for h in range(self.env.height):
            for w in range(self.env.width):
                pos = (h, w)
                for d in range(4):
                    possible_transitions = self.env.rail.get_transitions((pos, d))
                    if fast_count_nonzero(possible_transitions) > 1:
                        switches.setdefault(pos, []).append(d)

        switches_neighbours = {}
        for h in range(self.env.height):
            for w in range(self.env.width):
                pos = (h, w)
                for d in range(4):
                    possible_transitions = self.env.rail.get_transitions((pos, d))
                    for nd in range(4):
                        if possible_transitions[nd] == 1:
                            new_cell = get_new_position(pos, nd)
                            if new_cell in switches and pos not in switches:
                                switches_neighbours.setdefault(pos, []).append(d)

        self.switches = switches
        self.switches_neighbours = switches_neighbours

    def check_agent_decision(self, position, direction):
        agents_on_switch = False
        agents_near_to_switch = False
        agents_near_to_switch_all = False

        if position in self.switches:
            agents_on_switch = direction in self.switches[position]

        if position in self.switches_neighbours:
            new_cell = get_new_position(position, direction)
            if new_cell in self.switches:
                if direction not in self.switches[new_cell]:
                    agents_near_to_switch = direction in self.switches_neighbours[position]
            else:
                agents_near_to_switch = direction in self.switches_neighbours[position]
            agents_near_to_switch_all = direction in self.switches_neighbours[position]

        return agents_on_switch, agents_near_to_switch, agents_near_to_switch_all

    def _build_spawn_schedule(self):
        schedule = {}
        if self.env is None or self.env.agents is None:
            self._spawn_schedule = schedule
            return
        for h, agent in enumerate(self.env.agents):
            if agent.initial_position is not None:
                ed = agent.earliest_departure if agent.earliest_departure is not None else 0
                schedule.setdefault(agent.initial_position, []).append(
                    (ed, agent.initial_direction, h)
                )
        self._spawn_schedule = schedule

    def reset(self):
        self.build_data()
        self._build_spawn_schedule()
        self.waypoint_index = {}
        self._waypoint_distance_maps = {}
        if self.env is not None and self.env.rail is not None:
            for handle in range(self.env.get_num_agents()):
                self.waypoint_index[handle] = 0
                self._recompute_distance_map(handle)

    def _ensure_handle_initialized(self, handle):
        if not self._switches_built and self.env.rail is not None:
            self.find_all_cell_where_agent_can_choose()
            self._switches_built = True
        if not self._spawn_schedule and self.env is not None:
            self._build_spawn_schedule()
        if handle not in self.waypoint_index:
            self.waypoint_index[handle] = 0
            self._recompute_distance_map(handle)

    def _get_next_waypoint_target(self, handle):
        agent = self.env.agents[handle]
        wps = agent.waypoints
        next_idx = self.waypoint_index[handle] + 1
        if next_idx < len(wps):
            return wps[next_idx][0].position
        return agent.target

    def _compute_bfs_distance_map(self, target_position):
        rail = self.env.rail
        height, width = self.env.height, self.env.width
        dist = np.full((height, width, 4), np.inf)

        queue = deque()
        for d in range(4):
            if rail.get_transitions((target_position, d)) != (0, 0, 0, 0):
                dist[target_position[0], target_position[1], d] = 0
                queue.append((target_position, d, 0))

        while queue:
            pos, direction, current_dist = queue.popleft()
            prev_pos = get_new_position(pos, (direction + 2) % 4)
            if not (0 <= prev_pos[0] < height and 0 <= prev_pos[1] < width):
                continue
            for agent_dir in range(4):
                if rail.get_transition((prev_pos, agent_dir), direction):
                    new_dist = current_dist + 1
                    if new_dist < dist[prev_pos[0], prev_pos[1], agent_dir]:
                        dist[prev_pos[0], prev_pos[1], agent_dir] = new_dist
                        queue.append((prev_pos, agent_dir, new_dist))
        return dist

    def _recompute_distance_map(self, handle):
        target_pos = self._get_next_waypoint_target(handle)
        self._waypoint_distance_maps[handle] = self._compute_bfs_distance_map(target_pos)

    def _get_distance_map(self, handle):
        agent = self.env.agents[handle]
        if len(agent.waypoints) <= 2:
            return self.env.distance_map.get()[handle]
        return self._waypoint_distance_maps[handle]

    def _check_and_advance_waypoint(self, handle):
        self._ensure_handle_initialized(handle)
        agent = self.env.agents[handle]
        wps = agent.waypoints
        if len(wps) <= 2:
            return
        next_idx = self.waypoint_index[handle] + 1
        if next_idx >= len(wps) or agent.position is None:
            return
        for wp in wps[next_idx]:
            if wp.position == agent.position:
                self.waypoint_index[handle] = next_idx
                self._recompute_distance_map(handle)
                break

    def _get_remaining_dist(self, handle):
        agent = self.env.agents[handle]
        if agent.state == TrainState.DONE:
            return 0.0
        pos = agent.position if agent.position is not None else agent.initial_position
        direction = agent.direction if agent.direction is not None else agent.initial_direction
        if pos is None or direction is None:
            return float(self.env.width + self.env.height)
        dmap = self._get_distance_map(handle)
        d = dmap[pos[0], pos[1], direction]
        return float(d) if np.isfinite(d) else float(self.env.width + self.env.height)

    def _get_agent_slack(self, agent) -> float:
        la = agent.latest_arrival if agent.latest_arrival is not None else self.env._max_episode_steps
        return max(0.0, min(1.0, (la - self.env._elapsed_steps) / max(1, self.env._max_episode_steps)))

    def _get_malfunction_severity(self, agent) -> float:
        if agent.state != TrainState.MALFUNCTION:
            return 0.0
        mh = getattr(agent, "malfunction_handler", None)
        rem = getattr(mh, "malfunction_down_counter", 10) if mh is not None else 10
        return max(0.15, min(1.0, float(rem) / 50.0))

    def _check_future_spawn_conflict(self, handle, cell_pos, t_arrival, my_dir, my_dist):
        spawns = self._spawn_schedule.get(cell_pos)
        if not spawns:
            return False, False, 1.0

        for ed, spawn_dir, other_h in spawns:
            if other_h == handle or other_h >= len(self.env.agents):
                continue
            other = self.env.agents[other_h]
            if other.state != TrainState.READY_TO_DEPART:
                continue

            if (ed - 2.0) <= t_arrival <= (ed + 12.0):
                other_dist = self._get_remaining_dist(other_h)
                is_opp = (spawn_dir != my_dir)
                i_win = 1.0 if (my_dist < other_dist and my_dist < 10.0) else 0.0
                return True, is_opp, i_win

        return False, False, 1.0

    def _count_behind(self, handle, pos, direction, max_steps=20) -> int:
        """Scan backward along the track to count friendly trains queued behind this agent."""
        if pos is None or pos == (-1, -1):
            return 0
        count = 0
        num_agents = len(self.env.agents)
        curr_pos = pos
        curr_dir = (int(direction) + 2) % 4  # Look backward

        for _ in range(max_steps):
            pt = self.env.rail.get_transitions((curr_pos, curr_dir))
            if fast_count_nonzero(pt) != 1:
                break
            curr_dir = fast_argmax(pt)
            curr_pos = get_new_position(curr_pos, curr_dir)
            if not (0 <= curr_pos[0] < self.env.height and 0 <= curr_pos[1] < self.env.width):
                break
            occ = self.env.agent_positions[curr_pos]
            if 0 <= occ < num_agents and occ != handle:
                other = self.env.agents[occ]
                # Friendly train moving in the opposite of our backward scan (= our forward direction)
                if other.direction == (curr_dir + 2) % 4:
                    count += 1
        return count

    def _explore(self, handle, new_position, new_direction, distance_map, depth=0, steps_so_far=0):
        """
        Returns: (opp_r, same_r, stopped_same_r, malf_sev, esc_r, prio_win, congestion, opp_count, opp_slack_sum, visited)
        Full pass-through scan: counts total oncoming trains (platoon size) and same-direction queues.
        """
        if depth >= self.max_depth:
            return 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0, 0.0, []

        visited = []
        cnt = 0
        num_agents = len(self.env.agents)
        my_agent = self.env.agents[handle]
        my_speed = max(0.1, float(my_agent.speed_counter.speed))
        my_dist = self._get_remaining_dist(handle)
        now_step = float(self.env._elapsed_steps)

        max_opp_r = 0.0
        max_same_r = 0.0
        max_stopped_same_r = 0.0
        max_malf_sev = 0.0
        min_prio_win = 1.0
        congestion_cnt = 0.0
        opp_count = 0
        opp_slack_sum = 0.0

        while cnt < 80:
            cnt += 1
            total_steps = steps_so_far + cnt
            t_arrival = now_step + (total_steps / my_speed)
            visited.append(new_position)

            opp_a = self.env.agent_positions[new_position]
            if 0 <= opp_a < num_agents and opp_a != handle:
                other = self.env.agents[opp_a]
                prox = max(0.15, 1.0 - (total_steps / 80.0))
                malf_sev = self._get_malfunction_severity(other)
                if malf_sev > max_malf_sev:
                    max_malf_sev = malf_sev
                congestion_cnt += 1.0

                if other.direction != new_direction:
                    other_dist = self._get_remaining_dist(opp_a)
                    if my_agent.state == TrainState.READY_TO_DEPART:
                        i_win = 0.0
                    else:
                        i_win = 1.0 if (my_dist < other_dist or (my_dist == other_dist and handle < opp_a)) else 0.0
                    max_opp_r = max(max_opp_r, prox)
                    min_prio_win = min(min_prio_win, i_win)
                    opp_count += 1
                    opp_slack_sum += self._get_agent_slack(other)
                else:
                    max_same_r = max(max_same_r, prox)
                    if other.state in (TrainState.STOPPED, TrainState.MALFUNCTION):
                        max_stopped_same_r = max(max_stopped_same_r, prox)

            spawn_hit, spawn_opp, spawn_win = self._check_future_spawn_conflict(
                handle, new_position, t_arrival, new_direction, my_dist
            )
            if spawn_hit:
                prox = max(0.2, 1.0 - (total_steps / 80.0))
                congestion_cnt += 1.0
                if spawn_opp:
                    max_opp_r = max(max_opp_r, prox)
                    min_prio_win = min(min_prio_win, spawn_win)
                    opp_count += 1
                    opp_slack_sum += 0.5
                else:
                    max_same_r = max(max_same_r, prox)

            agents_on_switch, _, _ = self.check_agent_decision(
                new_position, new_direction
            )
            possible_transitions = self.env.rail.get_transitions(
                (new_position, new_direction)
            )
            if agents_on_switch:
                goal_branches = []
                all_branches = []
                for dir_loop in range(4):
                    if possible_transitions[dir_loop] == 1:
                        next_p = get_new_position(new_position, dir_loop)
                        next_d = distance_map[next_p[0], next_p[1], dir_loop]
                        res = self._explore(
                            handle, next_p, dir_loop, distance_map, depth + 1, total_steps
                        )
                        visited.extend(res[9])
                        all_branches.append(res)
                        if np.isfinite(next_d):
                            goal_branches.append((next_d, res))

                if goal_branches:
                    goal_branches.sort(key=lambda x: x[0])
                    best_res = goal_branches[0][1]
                    has_alt_free = (
                        1.0
                        if any(r[0] == 0.0 and r[3] == 0.0 and r[2] == 0.0 for _, r in goal_branches[1:])
                        else best_res[4]
                    )
                    return (
                        max(max_opp_r, best_res[0]),
                        max(max_same_r, best_res[1]),
                        max(max_stopped_same_r, best_res[2]),
                        max(max_malf_sev, best_res[3]),
                        has_alt_free,
                        min(min_prio_win, best_res[5]),
                        min(1.0, (congestion_cnt / 4.0) + best_res[6]),
                        opp_count + best_res[7],
                        opp_slack_sum + best_res[8],
                        visited,
                    )
                elif all_branches:
                    r0 = all_branches[0]
                    return (
                        max(max_opp_r, r0[0]),
                        max(max_same_r, r0[1]),
                        max(max_stopped_same_r, r0[2]),
                        max(max_malf_sev, r0[3]),
                        0.0,
                        min(min_prio_win, r0[5]),
                        min(1.0, (congestion_cnt / 4.0) + r0[6]),
                        opp_count + r0[7],
                        opp_slack_sum + r0[8],
                        visited,
                    )
                return (
                    max_opp_r, max_same_r, max_stopped_same_r, max_malf_sev,
                    0.0, min_prio_win, min(1.0, congestion_cnt / 4.0), opp_count, opp_slack_sum, visited,
                )
            else:
                if fast_count_nonzero(possible_transitions) == 0:
                    return (
                        max_opp_r, max_same_r, max_stopped_same_r, max_malf_sev,
                        0.0, min_prio_win, min(1.0, congestion_cnt / 4.0), opp_count, opp_slack_sum, visited,
                    )
                new_direction = fast_argmax(possible_transitions)
                new_position = get_new_position(new_position, new_direction)

        return (
            max_opp_r, max_same_r, max_stopped_same_r, max_malf_sev,
            1.0 if (max_opp_r == 0.0 and max_malf_sev == 0.0) else 0.0,
            min_prio_win, min(1.0, congestion_cnt / 4.0), opp_count, opp_slack_sum, visited,
        )

    def _get_neighbor_features(self, handle, pos, direction, my_slack):
        neighbors = np.zeros((self.NEIGHBOR_K, self.NEIGHBOR_DIM), dtype=np.float32)
        if pos is None or pos == (-1, -1):
            return neighbors.flatten()

        h_max = max(1, self.env.height)
        w_max = max(1, self.env.width)
        max_dist = float(h_max + w_max)
        candidates = []

        for other_h, other in enumerate(self.env.agents):
            if other_h == handle or other.position is None:
                continue
            if other.state not in (
                TrainState.MOVING,
                TrainState.STOPPED,
                TrainState.MALFUNCTION,
            ):
                continue
            dy = other.position[0] - pos[0]
            dx = other.position[1] - pos[1]
            manhattan = abs(dy) + abs(dx)
            if manhattan <= 40:
                candidates.append((manhattan, dy, dx, other_h, other))

        candidates.sort(key=lambda x: x[0])

        for idx, (_, dy, dx, other_h, other) in enumerate(candidates[: self.NEIGHBOR_K]):
            other_dir = (
                other.direction
                if other.direction is not None
                else other.initial_direction
            )
            is_opposite = 1.0 if ((other_dir + 2) % 4 == direction) else 0.0
            is_same_dir = 1.0 if (other_dir == direction) else 0.0
            is_stopped_or_malf = (
                1.0
                if other.state in (TrainState.STOPPED, TrainState.MALFUNCTION)
                else 0.0
            )
            other_slack = self._get_agent_slack(other)
            other_rem_dist = min(1.0, self._get_remaining_dist(other_h) / max_dist)

            neighbors[idx] = [
                dy / h_max,
                dx / w_max,
                is_opposite,
                is_same_dir,
                is_stopped_or_malf,
                other_slack - my_slack,
                other_rem_dist,
            ]

        return neighbors.flatten()

    def get(self, handle):
        self_obs = np.zeros(self.SELF_DIM, dtype=np.float32)
        visited = []
        agent = self.env.agents[handle]

        self._check_and_advance_waypoint(handle)

        agent_done = False
        if agent.state == TrainState.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
            self_obs[4] = 1.0
        elif agent.state in (
            TrainState.MOVING,
            TrainState.STOPPED,
            TrainState.MALFUNCTION,
        ):
            agent_virtual_position = agent.position
            self_obs[5] = 1.0
        else:
            self_obs[6] = 1.0
            agent_virtual_position = (-1, -1)
            agent_done = True

        direction = (
            agent.direction if agent.direction is not None else agent.initial_direction
        )
        max_dist = self.env.width + self.env.height
        current_cell_dist = np.inf
        branch_candidates = []
        raw_branch_dists = {}
        overall_priority_win = 1.0
        max_opp_risk = 0.0
        max_congestion = 0.0
        max_opp_count = 0
        total_opp_slack = 0.0
        total_opp_seen = 0

        stopped_same_slots = (13, 17, 21)

        if not agent_done:
            visited.append(agent_virtual_position)
            distance_map = self._get_distance_map(handle)
            current_cell_dist = distance_map[
                agent_virtual_position[0], agent_virtual_position[1], direction
            ]
            possible_transitions = self.env.rail.get_transitions(
                (agent_virtual_position, direction)
            )

            d_int = int(direction)
            action_dirs = [
                (d_int - 1) % 4,  # dir_loop=0 -> Action 1 (MOVE_LEFT)
                d_int,            # dir_loop=1 -> Action 2 (MOVE_FORWARD)
                (d_int + 1) % 4,  # dir_loop=2 -> Action 3 (MOVE_RIGHT)
            ]

            for dir_loop, branch_direction in enumerate(action_dirs):
                if possible_transitions[branch_direction]:
                    new_position = get_new_position(
                        agent_virtual_position, branch_direction
                    )
                    new_cell_dist = distance_map[
                        new_position[0], new_position[1], branch_direction
                    ]
                    raw_branch_dists[dir_loop] = new_cell_dist

                    (
                        opp_r, same_r, stopped_same_r, malf_sev,
                        esc_r, prio_win, cong, opp_cnt, opp_slk, v
                    ) = self._explore(
                        handle, new_position, branch_direction, distance_map, depth=0, steps_so_far=1
                    )
                    visited.extend(v)

                    self_obs[10 + dir_loop] = 1.0
                    self_obs[14 + dir_loop] = opp_r
                    self_obs[18 + dir_loop] = same_r
                    self_obs[stopped_same_slots[dir_loop]] = stopped_same_r
                    self_obs[22 + dir_loop] = esc_r
                    self_obs[36 + dir_loop] = malf_sev

                    if opp_r > max_opp_risk:
                        max_opp_risk = opp_r
                    if cong > max_congestion:
                        max_congestion = cong
                    if opp_cnt > max_opp_count:
                        max_opp_count = opp_cnt
                    total_opp_slack += opp_slk
                    total_opp_seen += opp_cnt
                    if opp_r > 0.0 and prio_win < 0.5:
                        overall_priority_win = 0.0

                    interference_penalty = (
                        (opp_r * 28.0)
                        + (opp_cnt * 8.0)
                        + (malf_sev * 35.0)
                        + (stopped_same_r * 22.0)
                        + (same_r * 5.0)
                        + (cong * 12.0)
                    )
                    eff_cost = (
                        new_cell_dist if np.isfinite(new_cell_dist) else max_dist * 2
                    ) + interference_penalty
                    branch_candidates.append((eff_cost, new_cell_dist, dir_loop))

            finite_dists = [d for d in raw_branch_dists.values() if np.isfinite(d)]
            min_b_dist = min(finite_dists) if finite_dists else 0.0
            for dir_loop, d in raw_branch_dists.items():
                if np.isfinite(d):
                    self_obs[26 + dir_loop] = min(1.0, max(0.0, (d - min_b_dist) / 30.0))
                else:
                    self_obs[26 + dir_loop] = 1.0

        self_obs[25] = max_congestion
        self_obs[29] = min(1.0, max_opp_count / 5.0)

        if branch_candidates:
            branch_candidates.sort(key=lambda x: x[0])
            best_dir_loop = branch_candidates[0][2]
            self_obs[best_dir_loop] = 1.0
            if len(branch_candidates) > 1:
                second_dir_loop = branch_candidates[1][2]
                self_obs[46] = float(second_dir_loop + 1) / 4.0
                cost_diff = max(
                    0.0,
                    min(1.0, (branch_candidates[1][0] - branch_candidates[0][0]) / 50.0),
                )
                self_obs[47] = cost_diff

        agents_on_switch, agents_near_to_switch, agents_near_to_switch_all = (
            self.check_agent_decision(agent_virtual_position, direction)
        )
        self_obs[7] = float(agents_on_switch)
        self_obs[8] = float(agents_near_to_switch)
        self_obs[9] = float(agents_near_to_switch_all)
        self_obs[39] = overall_priority_win

        behind_count = 0 if agent_done else self._count_behind(handle, agent_virtual_position, direction)
        self_obs[48] = min(1.0, behind_count / 5.0)
        # Platoon advantage: >0 means my group is larger than oncoming group; <0 means yield to larger oncoming platoon
        self_obs[49] = float(np.clip(((1 + behind_count) - max_opp_count) / 5.0, -1.0, 1.0))

        at_safe_hold = (
            agent.state == TrainState.READY_TO_DEPART
            or (agents_near_to_switch_all and not agents_on_switch)
        )
        platoon_loses = (1 + behind_count) < max_opp_count
        must_yield = 1.0 if (
            at_safe_hold and max_opp_risk > 0.25 and (platoon_loses or overall_priority_win < 0.5)
        ) else 0.0
        self_obs[3] = must_yield

        nb_active_agents = 0
        nb_done_agents = 0
        num_total = max(1, self.env.get_num_agents())
        for h in self.env.get_agent_handles():
            st = self.env.agents[h].state
            if st in (TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION):
                nb_active_agents += 1
            elif st == TrainState.DONE:
                nb_done_agents += 1

        self_obs[30] = (
            current_cell_dist / max_dist
            if (not agent_done and np.isfinite(current_cell_dist))
            else 1.0
        )
        self_obs[31] = self.env._elapsed_steps / max(1, self.env._max_episode_steps)
        self_obs[32] = nb_active_agents / num_total
        self_obs[33] = (
            max(0.0, 1.0 - current_cell_dist / max(1, max_dist))
            if (not agent_done and np.isfinite(current_cell_dist))
            else 0.0
        )

        wps = agent.waypoints
        n_intermediate = max(len(wps) - 2, 0)
        if n_intermediate > 0:
            served = min(max(0, self.waypoint_index[handle]), n_intermediate)
            self_obs[34] = served / n_intermediate
        else:
            self_obs[34] = 1.0

        next_idx = self.waypoint_index[handle] + 1
        wps_la = agent.waypoints_latest_arrival
        deadline = None
        if next_idx < len(wps_la) and wps_la[next_idx] is not None:
            deadline = wps_la[next_idx]
        elif agent.latest_arrival is not None:
            deadline = agent.latest_arrival

        my_slack = 1.0
        if deadline is not None:
            remaining = deadline - self.env._elapsed_steps
            my_slack = max(0.0, min(1.0, remaining / max(1, self.env._max_episode_steps)))
        self_obs[35] = my_slack

        if total_opp_seen > 0:
            avg_opp_slack = total_opp_slack / total_opp_seen
            self_obs[50] = float(np.clip(avg_opp_slack - my_slack, -1.0, 1.0))
        self_obs[51] = nb_done_agents / num_total

        self_obs[40] = float(agent.speed_counter.speed)
        self_obs[41] = self._get_malfunction_severity(agent)
        self_obs[42] = 1.0 if agent.state == TrainState.STOPPED else 0.0

        wps_ed = agent.waypoints_earliest_departure
        curr_idx = self.waypoint_index[handle]
        ed_wait = 0.0
        if curr_idx < len(wps_ed) and wps_ed[curr_idx] is not None:
            wait_steps = wps_ed[curr_idx] - self.env._elapsed_steps
            if wait_steps > 0:
                ed_wait = min(1.0, wait_steps / max(1, self.env._max_episode_steps))
        self_obs[43] = ed_wait

        on_stop = 0.0
        if not agent_done and agent.position is not None and 0 < curr_idx < len(wps) - 1:
            if any(wp.position == agent.position for wp in wps[curr_idx]):
                on_stop = 1.0
        self_obs[44] = on_stop
        self_obs[45] = min(1.0, len(wps) / 6.0)

        neighbor_obs = self._get_neighbor_features(
            handle,
            None if agent_done else agent_virtual_position,
            direction,
            my_slack,
        )

        full_obs = np.concatenate([self_obs, neighbor_obs])
        self.env.dev_obs_dict.update({handle: visited})

        if not self.with_action_mask:
            return full_obs

        mask = np.zeros(self.ACTION_MASK_SIZE, dtype=np.float32)
        mask[0] = 1.0  # DO_NOTHING
        if agent_done or agent.state in (TrainState.WAITING, TrainState.MALFUNCTION):
            pass
        elif agent.position is None:
            mask[2] = 1.0  # READY_TO_DEPART: DO_NOTHING(0) or MOVE_FORWARD(2)
        else:
            mask[4] = 1.0  # STOP_MOVING
            pt = self.env.rail.get_transitions((agent.position, agent.direction))
            d = int(agent.direction)
            mask[1] = float(pt[(d - 1) % 4])
            mask[2] = float(pt[d])
            mask[3] = float(pt[(d + 1) % 4])
            # Dead-end U-turn: Flatland requires MOVE_FORWARD(2) when only reverse transition exists
            if mask[1] == 0.0 and mask[2] == 0.0 and mask[3] == 0.0 and pt[(d + 2) % 4] == 1:
                mask[2] = 1.0

        return np.concatenate([full_obs, mask])

    @staticmethod
    def agent_can_choose(observation):
        return observation[7] == 1 or observation[8] == 1


MyObservationBuilder = EnhancedTreeObsBuilder