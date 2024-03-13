from flow.controllers.base_lane_changing_controller import BaseLaneChangeController
import random

class BL_LanceConroller(BaseLaneChangeController):
    """A lane-changing model used to move vehicles into lane 0."""

    def __init__(self, veh_id, lane_change_params=None):
        """Instantiate the base class for lane-changing controllers."""
        if lane_change_params is None:
            lane_change_params = {}

        self.veh_id = veh_id
        self.lane_change_params = lane_change_params

        self.last_lane_nums = 4


    ##### Below this is new code #####
    def get_lane_change_action(self, env):

        edge = env.k.vehicle.get_edge(self.veh_id)
        current_lane_nums = env.k.network.num_lanes(edge)
        now_lane = env.k.vehicle.get_lane(self.veh_id)

        if current_lane_nums > self.last_lane_nums:
            lc = random.randint(0, 1)

            if now_lane == 1:
                action = lc-1
            elif now_lane ==2:
                action = lc
        else:
            action = 0
        self.last_lane_nums = current_lane_nums

        return action
