import gymnasium as gym
from metadrive.envs import MetaDriveEnv
from metadrive.obs.image_obs import ImageStateObservation
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod
import numpy as np

class MyEnvNoTraffic(MetaDriveEnv):
    def __init__(self, config=None):
        super().__init__(config)
        self.prev_steering = 0.0
        self.prev_throttle = 0.0
        self.target_speed = 75.0  # km/h
        

    @classmethod
    def default_config(cls):
        config = super(MyEnvNoTraffic, cls).default_config()
        sensor_size = (80,60) 
        # Observation & Action Space
        config["agent_observation"] = ImageStateObservation
        config["discrete_action"] = False
        config["use_multi_discrete"] = False
        config["image_observation"] = True  
        # Map Configuration
        config["map"] = "SSSSCSSXCSSCCCCCCSCSSCCCSSSSSSSCSSSSSSSCSSS"
        config["start_seed"] = 0
        config["num_scenarios"] = 1
        config["map_config"].update({
            BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
            BaseMap.GENERATE_CONFIG: 30,
            BaseMap.LANE_WIDTH: 3.5,
            BaseMap.LANE_NUM: 4,
            "sidewalk_width": 1.0, 
            "exit_length": 50,
            "start_position": [0, 0],
        })

        # Traffic & Safety
        config["traffic_density"] = 0.0
        config["accident_prob"] = 0.0
        config["random_agent_model"] = False
        config["random_spawn_lane_index"] = False

        # Vehicle Sensors
        config["sensors"].update({
            "rgb": (RGBCamera, *(sensor_size)), # Sensor type and dimensions (W, H)
        })
        

        # Configure vehicle-specific settings, including which sensor to use for image obs
        config["vehicle_config"].update({
            "image_source": "rgb", 
        })

        # Reward Parameters
        config.update({
            "success_reward": 1000.0,
            "crash_penalty": 300.0,
            "out_of_road_penalty": 700.0,
            "speed_reward_coeff": 0.1,
            "heading_penalty_coeff": 0.15,
            "lateral_penalty_coeff": 0.2,
            "comfort_penalty_coeff": 0.05,
            "corner_penalty_coeff": 0.1,
            "time_penalty": 0.01,
            "driving_reward": 1.0,
        })

        # Termination Conditions
        config.update({
            "crash_vehicle_done": False,
            "crash_object_done": False,
            "out_of_road_done": True,
            "on_continuous_line_done": True,
            "horizon": 10_000,
        })

        return config

    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = {}

        # Route & Lane Tracking
        current_lane = vehicle.navigation.current_ref_lanes[0]
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)
        long_last,_ = current_lane.local_coordinates(vehicle.last_position)
        positive_road = 1 if vehicle.lane in vehicle.navigation.current_ref_lanes else -1

        # Base Rewards
        reward = 0.0

        # 1. Speed Control (Target Speed Reward + Safe Speed Penalty)
        speed_km_h = vehicle.speed_km_h
        speed_diff = abs(speed_km_h - self.target_speed)
        reward += self.config["speed_reward_coeff"] * (1 - speed_diff / self.target_speed)
        # penilizing imobility
        if speed_km_h < 0.5:
            reward-=1
        # Penalty for exceeding safe speed (e.g., during turns)
        if speed_km_h > self.target_speed:
            reward -= 0.2 * (speed_km_h - self.target_speed)

        # 2. Progress Along Route
        reward += self.config["driving_reward"] * (long_now - long_last) * positive_road

        # # 3. Lane Centering
        # lateral_penalty = -self.config["lateral_penalty_coeff"] * (abs(lateral_now) / current_lane.width)
        # reward += lateral_penalty

        # 4. Heading Alignment
        heading_diff = vehicle.heading_diff(current_lane)
        reward -= self.config["heading_penalty_coeff"] * abs(heading_diff)

        # 5. Comfort Penalty (Smooth Steering/Throttle)
        steering = vehicle.steering
        throttle = vehicle.throttle_brake
        steering_diff = abs(steering - self.prev_steering)
        throttle_diff = abs(throttle - self.prev_throttle)
        comfort_penalty = -self.config["comfort_penalty_coeff"] * (steering_diff + throttle_diff)
        reward += comfort_penalty
        self.prev_steering, self.prev_throttle = steering, throttle

        
        # 7. Termination Penalties
        if self._is_arrive_destination(vehicle):
            reward += self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward -= self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle or vehicle.crash_object:
            reward -= self.config["crash_penalty"]
            step_info["crash_vehicle"] = vehicle.crash_vehicle

        # 8. Time Penalty (Encourage efficiency)
        reward -= self.config["time_penalty"]

        step_info["step_reward"] = reward
        step_info["route_completion"] = vehicle.navigation.route_completion
        return reward, step_info