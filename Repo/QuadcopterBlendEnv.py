import numpy as np
from numpy.random import default_rng

import gym
from gym import spaces

import quadcopter
import Controller
from Helpers_Generic import set_random_straight_line_path, set_random_v_shaped_path
from RewardFunctions import ParameterizedDiscreteReward


class QuadcopterBlendEnv(gym.Env):
    """ Custom Gym environment for Quadcopter control.
    The agent must generate distribution parameters for the Randomized Blended Control
    of a Quadcopter.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, operating_modes=None, action_space_type='Default', verbose=False, switching_logic='Free',
                 reward_params=None, trajectory='Straight'):
        """
        Parameters
        ----------
        switching_logic : string
            Options are: 'Free', 'Boolean', 'Clip'.
        """
        # __init__ attribute definition
        self.cumu_reward = 0.0
        self.stable_at_goal = 0
        self.current_waypoint = 0
        self.total_waypoints = 0
        self.safe_region = []
        self.goals = []
        self.quad = None
        self.ctrl = None
        self.weight_log = []

        # Initializing and allocating
        self.verbose = verbose
        self.operating_modes = {}
        self.no_modes = 0
        self.modes_list = []
        self.current_mode = ''

        # Trajectory Type
        self.trajectory_type = trajectory
        if trajectory == 'Straight':
            self.max_control_steps = 5000
        elif trajectory == 'V-shaped':
            self.max_control_steps = 7000
        else:
            print(f"'{trajectory}' is not a compatible trajectory type.")

        # Reward
        self.mode_match_ranges = {'Nominal': [0.0, 0.8],
                                  'LOE_med': [0.0, 0.5],
                                  'LOE_high': [0.0, 0.2],
                                  'AttN_med': [0.5, 1.0],
                                  'AttN_high': [0.8, 1.0]}
        self.threshold_bumpless_reward = 0.2
        if reward_params is None:
            reward_params = {'outofbounds': 1,
                             'finalgoal': 500,
                             'timepenalty': 0.1,
                             'withinbounds': 0,
                             'matchenv': 0,
                             'bumpless': 0}
        self.reward_function = ParameterizedDiscreteReward(reward_params)

        # Operating modes
        if operating_modes is None:  # Default values
            operating_modes = {'Rotor': [0.0, 0.30],
                               'AttNoise': [0.0, 1.2]}
        self.set_operating_modes(operating_modes)

        # Switching logic
        self.weight_clip_range = 0.1  # Weight clip range for inside bounds. switching_logic = 'Clip'
        self.threshold_switch_logic = 0.5  # Bounds to start resampling control. switching_logic != 'Free'
        self.switching_logic = switching_logic

        # Quadcopter
        self.reset()

        # Action Space
        if action_space_type == 'Default':
            self.actionlow = 0.01
            self.actionhigh = 1
            self.action_low_state = np.array([self.actionlow, self.actionlow], dtype=np.float)
            self.action_high_state = np.array([self.actionhigh, self.actionhigh], dtype=np.float)
            self.action_space = spaces.Box(low=self.action_low_state, high=self.action_high_state, dtype=np.float)
        elif action_space_type == 'ContinuousRestricted':  # Continuous with restricted std
            mean_low = 0
            mean_high = 1
            std_low = 0.01
            std_high = 0.2
            self.action_low_state = np.array([mean_low, std_low], dtype=np.float)
            self.action_high_state = np.array([mean_high, std_high], dtype=np.float)
            self.action_space = spaces.Box(low=self.action_low_state, high=self.action_high_state, dtype=np.float)
        else:
            print(f"'{action_space_type}' is not a compatible action space type.")

        # Observation Space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,))

    def reset(self):
        # Quadcopter
        quad_weight = 1.2  # weight = default_rng.choice([0.8, 1.2, 1.6])
        quad_config = {str('1'): {'position': [0, 0, 5], 'orientation': [0, 0, 0], 'L': 0.3, 'r': 0.1,
                                  'prop_size': [10, 4.5], 'weight': quad_weight}}
        self.quad = quadcopter.Quadcopter(quad_config)

        # Controller
        controller_config = {
            'Motor_limits': [0, 9000],
            'Tilt_limits': [-10, 10],
            'Yaw_Control_Limits': [-900, 900],
            'Z_XY_offset': 500,
            'Linear_PID': {'P': [300, 300, 7000], 'I': [0.04, 0.04, 4.5], 'D': [450, 450, 5000]},
            'Linear_To_Angular_Scaler': [1, 1, 0],
            'Yaw_Rate_Scaler': 0.18,
            'Angular_PID': {'P': [24000, 24000, 1500], 'I': [0, 0, 1.2], 'D': [12000, 12000, 0]},
            'Angular_PID2': {'P': [4000, 4000, 1500], 'I': [0, 0, 1.2], 'D': [1500, 1500, 0]},
             }
        self.ctrl = Controller.Blended_PID_Controller(self.quad.get_state, self.quad.set_motor_speeds,
                                                      self.quad.get_motor_speeds, self.quad.stepQuad,
                                                      self.quad.set_motor_faults, self.quad.setWind,
                                                      self.quad.setNormalWind, params=controller_config)
        self.ctrl.set_controller("Agent")

        # Desired Trajectory
        if self.trajectory_type == 'Straight':
            self.goals, self.safe_region, self.total_waypoints = set_random_straight_line_path()
        elif self.trajectory_type == 'V-shaped':
            self.goals, self.safe_region, self.total_waypoints = set_random_v_shaped_path()

        # Initializing
        self.current_waypoint = 0
        self.stable_at_goal = 0
        self.ctrl.update_target(self.goals[self.current_waypoint], self.safe_region[self.current_waypoint])
        self.cumu_reward = 0
        self._randomize_mode()
        self.weight_log = []

        obs = self.ctrl.get_observations()
        return obs

    def step(self, action):
        """Action will be probability distribution parameters for blend space
        Parameters
        ----------
        action : ndarray/list
            [mean, std]
        """
        # Init
        info = {}
        done = False
        rew_time_penalty = 0
        rew_goal = 0
        weight_clip = None

        # Switching logic
        if self.switching_logic != 'Free':  # No free blending
            if not self.weight_log:  # First step. No previous weight
                pass  # Continue to step with free blending
            else:
                out_of_switch_threshold = self.ctrl.get_distance_to_opt() > self.threshold_switch_logic
                prev_weight = self.weight_log[-1]

                if self.switching_logic == 'Boolean':
                    if out_of_switch_threshold:  # Out of switch threshold?
                        weight_clip = None  # Blend control freely.
                    else:  # Within switch threshold?
                        weight_clip = [prev_weight, prev_weight]  # Forcing previous weight through clipping.
                elif self.switching_logic == 'Clip':
                    if out_of_switch_threshold:  # Out of switch threshold?
                        weight_clip = None  # Blend control freely.
                    else:  # Within switch threshold?
                        weight_clip = [prev_weight - self.weight_clip_range, prev_weight + self.weight_clip_range]

        # Step Control
        obs, weight = self.ctrl.set_action(action, weight_clip)

        self.weight_log.append(weight)

        if self.ctrl.total_steps > self.max_control_steps:
            done = True
            rew_time_penalty = -1

        if self.ctrl.is_at_pos(self.goals[self.current_waypoint]):  # If the controller has reached the waypoint
            if self.current_waypoint < self.total_waypoints - 1:  # If not final goal
                self.current_waypoint += 1  # Next waypoint
                self.ctrl.update_target(self.goals[self.current_waypoint], self.safe_region[self.current_waypoint - 1])
            else:  # If final goal
                self.stable_at_goal += 1  # Number of timesteps spent within final goal
                if self.stable_at_goal > 20:
                    done = True
                    rew_goal = 1
        else:
            self.stable_at_goal = 0

        # Reward
        # Out of bounds
        rew_out_of_bounds = self.ctrl.get_reward()
        # Environment-control match
        mode_range = self.mode_match_ranges[self.current_mode]
        if mode_range[0] <= weight <= mode_range[1]:
            rew_env_match = 0
        else:
            rew_env_match = -1
        # Bumpless control
        if len(self.weight_log) < 2:  # First step. No previous weight
            rew_bumpless = 0
        else:
            weight_change = abs(weight - self.weight_log[-2])  # Difference between current and previous weight.
            if weight_change < self.threshold_bumpless_reward:
                rew_bumpless = 0
            else:
                rew_bumpless = -1
        # Reward Dictionary
        reward_tags = {'outofbounds': rew_out_of_bounds,
                       'finalgoal': rew_goal,
                       'timepenalty': rew_time_penalty,
                       'withinbounds': 0,
                       'matchenv': rew_env_match,
                       'bumpless': rew_bumpless}
        reward = self.reward_function.calculate_reward(reward_tags)
        self.cumu_reward = self.cumu_reward + reward

        # Done
        if done:
            # Do reward things
            info = {"Cumulative Reward": self.cumu_reward}

        return np.array(obs), reward, done, info

    def close(self):
        pass

    def render(self, mode='human'):
        pass

    def set_operating_modes(self, operating_modes_dict):
        """Setter method for operating modes dict

        Parameters
        ----------
        operating_modes_dict: dict
            Dictionary holding operating modes and their ranges.

        """
        self.operating_modes = operating_modes_dict
        self.no_modes = len(self.operating_modes)
        self.modes_list = sorted(self.operating_modes)  # Sorting the key list guarantees consistency

    def _randomize_mode(self):
        """ Set the environment in a randomly generated operating mode

        """
        # Selecting random mode from the available operating modes
        rng = default_rng()
        rand_mode = rng.integers(0, self.no_modes)
        rand_key = self.modes_list[rand_mode]
        self.ctrl.set_fault_mode(rand_key)

        # Setting up selected mode
        if rand_key == 'Nominal':
            self.current_mode = 'Nominal'
            # Verbose
            if self.verbose:
                print('Operating Mode: ', rand_key)  # No changes needed for nominal conditions
        elif rand_key == 'Rotor':
            # Randomly generating anomaly magnitude
            rand_coeff = rng.uniform()  # **** Consider including ranges
            # Getting loe range from operating modes dict
            mode_range = self.operating_modes[rand_key]
            fault_mag = (mode_range[1] - mode_range[0]) * rand_coeff + mode_range[0]
            # Setting LOE values on a single rotor
            faults = [0, 0, 0, 0]
            rotor = np.random.randint(0, 4)  # Random Rotor Selection
            faults[rotor] = fault_mag
            starttime = 1
            endtime = 31000
            self.ctrl.set_motor_fault(faults)
            self.ctrl.set_fault_time(starttime, endtime)
            # Current Mode
            if fault_mag < 0.1:
                self.current_mode = 'Nominal'
            elif fault_mag < 0.2:
                self.current_mode = 'LOE_med'
            else:
                self.current_mode = 'LOE_high'
            # Verbose
            if self.verbose:
                print('Operating Mode: ', rand_key)
                print('LOE (%): ', fault_mag)
        elif rand_key == 'AttNoise':
            # Randomly generating anomaly magnitude
            rand_coeff = rng.uniform()  # **** Consider including ranges
            # Getting position noise range from operating modes dict
            mode_range = self.operating_modes[rand_key]
            fault_mag = (mode_range[1] - mode_range[0]) * rand_coeff + mode_range[0]
            self.ctrl.set_attitude_sensor_noise(fault_mag)
            # Current Mode
            if fault_mag < 0.4:
                self.current_mode = 'Nominal'
            elif fault_mag < 0.8:
                self.current_mode = 'AttN_med'
            else:
                self.current_mode = 'AttN_high'
            # Verbose
            if self.verbose:
                print('Operating Mode: ', rand_key)
                print('Attitude Noise (rad): ', fault_mag)

