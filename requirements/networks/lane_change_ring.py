"""Contains the ring road scenario class."""

from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from numpy import pi, sin, cos, linspace
import numpy as np

ADDITIONAL_NET_PARAMS = {
    # length of the ring road
    "length": 230,
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 30,
    # resolution of the curves on the ring
    "resolution": 40
}


class RingNetwork(Network):
    """Ring road network.

    This network consists of nodes at the top, bottom, left, and right
    peripheries of the circles, connected by four 90 degree arcs. It is
    parametrized by the length of the entire network and the number of lanes
    and speed limit of the edges.

    Requires from net_params:

    * **length** : length of the circle
    * **lanes** : number of lanes in the circle
    * **speed_limit** : max speed limit of the circle
    * **resolution** : number of nodes resolution

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.networks import RingNetwork
    >>>
    >>> network = RingNetwork(
    >>>     name='ring_road',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams(
    >>>         additional_params={
    >>>             'length': 230,
    >>>             'lanes': 1,
    >>>             'speed_limit': 30,
    >>>             'resolution': 40
    >>>         },
    >>>     )
    >>> )
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize a ring scenario."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        length = net_params.additional_params["length"]
        r = length / (2 * pi)

        nodes = [{
            "id": "bottom",
            "x": 0,
            "y": -r
        }, {
            "id": "right",
            "x": r,
            "y": 0
        }, {
            "id": "top",
            "x": 0,
            "y": r
        }, {
            "id": "left",
            "x": -r,
            "y": 0
        }]

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        length = net_params.additional_params["length"]
        resolution = net_params.additional_params["resolution"]
        r = length / (2 * pi)
        edgelen = length / 4.

        edges = [{
            "id":
                "bottom",
            "type":
                "edgeType",
            "from":
                "bottom",
            "to":
                "right",
            "length":
                edgelen,
            "shape":
                [
                    (r * cos(t), r * sin(t))
                    for t in linspace(-pi / 2, 0, resolution)
                ]
        }, {
            "id":
                "right",
            "type":
                "edgeType",
            "from":
                "right",
            "to":
                "top",
            "length":
                edgelen,
            "shape":
                [
                    (r * cos(t), r * sin(t))
                    for t in linspace(0, pi / 2, resolution)
                ]
        }, {
            "id":
                "top",
            "type":
                "edgeType",
            "from":
                "top",
            "to":
                "left",
            "length":
                edgelen,
            "shape":
                [
                    (r * cos(t), r * sin(t))
                    for t in linspace(pi / 2, pi, resolution)
                ]
        }, {
            "id":
                "left",
            "type":
                "edgeType",
            "from":
                "left",
            "to":
                "bottom",
            "length":
                edgelen,
            "shape":
                [
                    (r * cos(t), r * sin(t))
                    for t in linspace(pi, 3 * pi / 2, resolution)
                ]
        }]

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        lanes = net_params.additional_params["lanes"]
        speed_limit = net_params.additional_params["speed_limit"]

        types = [{
            "id": "edgeType",
            "numLanes": lanes,
            "speed": speed_limit
        }]

        return types

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "top": ["top", "left", "bottom", "right"],
            "left": ["left", "bottom", "right", "top"],
            "bottom": ["bottom", "right", "top", "left"],
            "right": ["right", "top", "left", "bottom"]
        }

        return rts

    def specify_edge_starts(self):
        """See parent class."""
        ring_length = self.net_params.additional_params["length"]
        junction_length = 0.1  # length of inter-edge junctions

        edgestarts = [("bottom", 0),
                      ("right", 0.25 * ring_length + junction_length),
                      ("top", 0.5 * ring_length + 2 * junction_length),
                      ("left", 0.75 * ring_length + 3 * junction_length)]

        return edgestarts

    def specify_internal_edge_starts(self):
        """See parent class."""
        ring_length = self.net_params.additional_params["length"]
        junction_length = 0.1  # length of inter-edge junctions

        edgestarts = [(":right_0", 0.25 * ring_length),
                      (":top_0", 0.5 * ring_length + junction_length),
                      (":left_0", 0.75 * ring_length + 2 * junction_length),
                      (":bottom_0", ring_length + 3 * junction_length)]

        return edgestarts

    @staticmethod
    def gen_custom_start_pos(cls, net_params, initial_config, num_vehicles):
        VEHICLE_LENGTH: int = 5

        length = net_params.additional_params['length']
        num_lane = net_params.additional_params['lanes']
        min_gap = initial_config.min_gap
        my_min_gap = min_gap + 23
        startpos, startlane = [], []

        if length < num_vehicles * (VEHICLE_LENGTH + my_min_gap):
            raise ValueError('num of vehicles are too many')

        surplus_gap = length - num_vehicles * (VEHICLE_LENGTH + my_min_gap)
        tmp = list(range(int(num_vehicles//2+num_vehicles%2))) + list(reversed(range(int(num_vehicles//2))))
        surplus_gap_list = np.array(tmp)/sum(tmp) * surplus_gap

        edges = ['bottom', 'right', 'top', 'left']
        edge_length = length // 4

        startpos.append((edges[0], 0))
        startlane.append(0)

        lane_index = list(range(0,num_lane, 2))+list(range(1, num_lane, 2))
        startlane += [lane_index[i%len(lane_index)] for i in range(num_vehicles-1)]

        pos = 0
        for veh, gap in zip(range(num_vehicles - 1), surplus_gap_list):
            pos = pos + my_min_gap + VEHICLE_LENGTH + gap
            startpos.append((edges[int(pos // edge_length)], pos % edge_length))

        else:
            rl_pos = startpos.pop(0)
            rl_lane = startlane.pop(0)
            startpos.append(rl_pos)
            startlane.append(rl_lane)

        return startpos, startlane

    @staticmethod
    def gen_custom_start_pos2(cls, net_params, initial_config, num_vehicles):
        VEHICLE_LENGTH = 5

        length = net_params.additional_params['length']
        num_lane = net_params.additional_params['lanes']
        min_gap = initial_config.min_gap
        my_min_gap = min_gap + 20
        startpos, startlane = [], []
        if length < (num_vehicles / num_lane) * (VEHICLE_LENGTH + my_min_gap):
            raise ValueError('num of vehicles are too many')

        surplus_gap = length - num_vehicles * (VEHICLE_LENGTH + my_min_gap)
        tmp = list(range(int(num_vehicles // 2 + num_vehicles % 2))) + list(reversed(range(int(num_vehicles // 2))))
        surplus_gap_list = np.array(tmp) / sum(tmp) * surplus_gap
        avg_gap = np.mean(surplus_gap_list)
        edges = ['bottom', 'right', 'top', 'left']
        edge_length = length // 4

        lane_index = list(range(0, num_lane, 3)) + list(range(1, num_lane, 3)) + list(range(2, num_lane, 3))
        startlane += [lane_index[i % len(lane_index)] for i in range(num_vehicles - 1)]

        pos = 0
        pos_list = []
        for veh in range(num_vehicles):
            if veh % 2 == 1:
                pos = pos + 3.2 * (my_min_gap + VEHICLE_LENGTH)

            if veh % 2 == 0:
                pos = pos

            pos_list.append(pos)
            startpos.append((edges[int(pos // edge_length)], pos % edge_length))

        startlane.append(0)

        return startpos, startlane