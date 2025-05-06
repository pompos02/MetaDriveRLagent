import gymnasium as gym
from metadrive.envs import MetaDriveEnv
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod
from numpy import clip

class StraightEnv(MetaDriveEnv):
    @classmethod
    def default_config(cls):
        config = super(StraightEnv, cls).default_config()

        config["agent_observation"] = LidarStateObservation

        config["map"] = "SSSSSSSSS"

        config["traffic_density"] = 0.0

        config["random_agent_model"] = False

        config["discrete_action"] = False
        config["use_multi_discrete"] = False

        config["horizon"] = 4096
        config["map_config"].update({
            BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
            BaseMap.GENERATE_CONFIG: None,  
            BaseMap.LANE_WIDTH: 3.5,
            BaseMap.LANE_NUM: 4,
            "exit_length": 50,
            "start_position": [0, 0],
        })

        config["vehicle_config"].update({
            "lidar": {
                "num_lasers": 72,
                "distance": 50,
                "num_others": 4,
                "add_others_navi": False,
                "gaussian_noise": 0.0,
                "dropout_prob": 0.0,
            },
            "side_detector": {
                "num_lasers": 20,
                "distance": 5,
            },
            "lane_line_detector": {
                "num_lasers": 20,
                "distance": 5,
            },
            "show_lidar": False,
            "show_side_detector": False,
            "show_lane_line_detector": False,
        })

        config["success_reward"]= 100.0
        config["crash_vehicle_penalty"]= 50.0
        config["crash_object_penalty"]= 50.0
        config["crash_building_penalty"]= 50.0
        config["out_of_road_penalty"]= 50.0
        config["driving_reward"]= 1.0
        config["speed_reward"]= 1.0

        config["use_lateral_reward"] = False
        config["out_of_route_done"] = False
        
        return config
    
    

    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()

        # Reward for keeping the same lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last,_ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)





        #print(self.config)
        reward = 0.0
        reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) 
        reward += self.config["driving_reward"]*(long_now - long_last) * positive_road

        # MAX_SAFE_SPEED = 40.0  # km/h
        # current_speed = vehicle.speed_km_h
        # print("Current speed: ", current_speed)
        # if current_speed > MAX_SAFE_SPEED:
        #     overshoot = current_speed - MAX_SAFE_SPEED
        #     # Subtract more penalty the faster it goes beyond 40 km/h
        #     reward -= 0.2 * overshoot



        if self._is_arrive_destination(vehicle):
            reward += self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward += -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward += -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward += -self.config["crash_object_penalty"]
        elif vehicle.crash_sidewalk:
            reward += -self.config["crash_sidewalk_penalty"]
        step_info["route_completion"] = vehicle.navigation.route_completion
        step_info["speed"] = vehicle.speed_km_h
        
        step_info["step_reward"] = reward
        return reward, step_info
    
