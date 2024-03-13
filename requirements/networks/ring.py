"""Contains the ring road scenario class."""

from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from numpy import pi, sin, cos, linspace

import math
import copy
import random

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
        import time

        VEHICLE_LENGTH = 4.45
        NO_LEFT_VEHICLE_NUMS = 36

        length = net_params.additional_params['length']
        num_lane = net_params.additional_params['lanes']
        min_gap = initial_config.min_gap
        my_min_gap = min_gap + 28
        running_start_pos=0

        no_leftside_veh_tuple = []
        other_veh_tuple = []

        startpos, startlane = [], []
        available_pos_tuple =[]

        if length*num_lane < (my_min_gap)*num_vehicles:
            raise ValueError('num of vehicles are too many')

        edges = ['bottom', 'right', 'top', 'left']
        edge_length = length // 4

        for i in range(math.ceil(num_vehicles/num_lane)):
            available_pos_tuple.append((edges[int(running_start_pos // edge_length)], running_start_pos%edge_length))
            running_start_pos+=my_min_gap

        unplaced_veh = num_vehicles
        for edge, pos in available_pos_tuple:
            for lane in range(num_lane-1):
                now_pos = pos+round(random.uniform(0,my_min_gap/2), 1)
                if now_pos > edge_length:
                    real_edge_index = edges.index(edge)
                    edge = edges[real_edge_index]
                no_leftside_veh_tuple.append(((edge, now_pos), lane))
                unplaced_veh-=1

        while unplaced_veh!=0:
            other_veh_tuple.append((available_pos_tuple[unplaced_veh], num_lane-1))
            unplaced_veh-=1

        random.shuffle(no_leftside_veh_tuple)
        other_veh_tuple

        must_keep_lane_veh = copy.deepcopy(no_leftside_veh_tuple[:NO_LEFT_VEHICLE_NUMS])
        final_other_vehicle = copy.deepcopy(no_leftside_veh_tuple[NO_LEFT_VEHICLE_NUMS:])+other_veh_tuple

        random.shuffle(final_other_vehicle)

        final_tuple = must_keep_lane_veh+final_other_vehicle
        startpos, startlane = zip(*final_tuple)

        return startpos, startlane