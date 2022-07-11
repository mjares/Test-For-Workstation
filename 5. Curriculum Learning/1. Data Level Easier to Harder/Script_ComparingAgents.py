import os
import sys
from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Adding local libs to path
parentFolder = Path(os.getcwd()).parent
sys.path.append(str(parentFolder) + '\\Z. Local mods')

# Local libs
import quadcopter
import Controller
from Helpers_Generic import set_fixed_square_path, plot_3d_trajectory
from HildensiaDataset import HildensiaDataset

### Script ###

no_datasets = 50
mu = 0.5
sigma = 0.5
switching_logic = 'Free'
reward_type = 'Match'
weight_clip_range = 0.1  # Weight clip range for inside bounds. switching_logic = 'Clip'
threshold_switch_logic = 0.5  # Bounds to start resampling control. switching_logic != 'Free'

datafolder = 'Agents'
filelist = os.listdir(datafolder)

for filename in filelist:
    dataset_list = []
    modelname = filename[:-4]
    print(modelname)
    for ii in range(no_datasets):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        rotor_mag = 0.30
        att_mag = 1.0
        domain = 1  # 1 LOE 4 Att Noise
        total_steps = []
        starttime = 1300
        endtime = 31000
        trajectories = []
        quad_id = 1

        # Make objects for quadcopter
        QUADCOPTER = {
            str(quad_id): {'position': [0, 0, 0], 'orientation': [0, 0, 0], 'L': 0.3, 'r': 0.1, 'prop_size': [10, 4.5],
                           'weight': 1.2}}
        quad = quadcopter.Quadcopter(QUADCOPTER)
        Quads = {str(quad_id): QUADCOPTER[str(quad_id)]}

        # Create blended controller and link it to quadcopter object
        BLENDED_CONTROLLER_PARAMETERS = {
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

        model = PPO.load('Agents/' + modelname)
        ctrl = Controller.Blended_PID_Controller(quad.get_state,
                                                 quad.set_motor_speeds, quad.get_motor_speeds,
                                                 quad.stepQuad, quad.set_motor_faults, quad.setWind, quad.setNormalWind,
                                                 params=BLENDED_CONTROLLER_PARAMETERS)
        ctrl.set_controller("Agent")
        goals, safe_region, total_waypoints = set_fixed_square_path()
        currentWaypoint = 0
        ctrl.update_target(goals[currentWaypoint], safe_region[currentWaypoint])

        # Faults
        faultType = "None"
        if domain == 1:
            faultType = "Rotor"
            faults = [0, 0, 0, 0]
            rotor = np.random.randint(0, 4)  # Random Rotor Selection
            fault_mag = rotor_mag
            log_mag = rotor_mag
            faults[rotor] = fault_mag
            ctrl.set_motor_fault(faults)
            ctrl.set_fault_time(starttime, endtime)  # Fixed Start time
        elif domain == 4:
            faultType = "AttNoise"
            ctrl.set_fault_time(0, 310000)  # Fixed Start time
            ctrl.set_attitude_sensor_noise(att_mag)
            log_mag = att_mag
        else:
            log_mag = 0
        ctrl.set_fault_mode(faultType)

        # Trajectory Tracking Initialization
        obs, _ = ctrl.set_action([mu, sigma])
        done = False
        stepcount = 0
        stableAtGoal = 0
        obs_log = []
        weight_log = []
        reward_log = []
        action_log = []
        error_log = []
        goal_log = []
        stable_done = False

        while not done:
            stepcount += 1

            # Switching logic
            weight_clip = None
            if switching_logic != 'Free':  # No free blending
                if not weight_log:  # First step. No previous weight
                    pass  # Continue to step with free blending
                else:
                    out_of_switch_threshold = ctrl.get_distance_to_opt() > threshold_switch_logic  # True if switch.
                    prev_weight = weight_log[-1]

                    if switching_logic == 'Boolean':
                        if out_of_switch_threshold:  # Out of switch threshold?
                            weight_clip = None  # Blend control freely.
                        else:  # Within switch threshold?
                            weight_clip = [prev_weight, prev_weight]  # Forcing previous weight through clipping.
                    elif switching_logic == 'Clip':
                        if out_of_switch_threshold:  # Out of switch threshold?
                            weight_clip = None  # Blend control freely.
                        else:  # Within switch threshold?
                            weight_clip = [prev_weight - weight_clip_range, prev_weight + weight_clip_range]

            action, _ = model.predict(obs)
            obs, weight = ctrl.set_action(action, weight_clip)
            reward = ctrl.get_reward()

            # Logs
            obs_log.append(obs)
            action_log.append(action)
            weight_log.append(weight)
            reward_log.append(reward)
            goal_log.append(goals[currentWaypoint])
            error_log.append(ctrl.get_distance_to_opt())

            if stepcount > 10000:  # Max stepcount reached
                done = True
                trajectories = ctrl.get_trajectory()
                total_steps.append(ctrl.get_total_steps())

            if ctrl.is_at_pos(goals[currentWaypoint]):  # If the controller has reached the waypoint
                if currentWaypoint < total_waypoints - 1:  # If not final goal
                    currentWaypoint += 1  # Next waypoint
                    ctrl.update_target(goals[currentWaypoint], safe_region[currentWaypoint-1])
                else:  # If final goal
                    stableAtGoal += 1  # Number of timesteps spent within final goal
                    if stableAtGoal > 100:
                        stable_done = True
                        done = True
                        trajectories = ctrl.get_trajectory()
                        total_steps.append(ctrl.get_total_steps())
            else:
                stableAtGoal = 0

        ### Plotting
        trajectories = np.array(trajectories)
        # plot_3d_trajectory(trajectories, safe_region, faultType, show=False)
        # plt.figure()
        # plt.plot(weight_log)
        ## To Hildensia Dataset
        dataset = HildensiaDataset()
        dataset.partition = [starttime, stepcount]
        dataset.observations = np.array(obs_log)
        dataset.actions = np.array(action_log)
        dataset.reward = np.array(reward_log)
        dataset.weight = np.array(weight_log)
        dataset.safezone_error = np.array(error_log)
        dataset.goal = np.array(goal_log)
        dataset.safe_region = safe_region
        dataset.trajectories = trajectories
        dataset.conditions = f'Fault: {faultType}. Mag: {log_mag}'
        dataset.stable_at_goal = stable_done
        dataset_list.append(dataset)

        if stable_done:
            print('Stable at goal')
        else:
            print('Not stable at goal')
    # plt.show()
    saveFilename = 'Hildensia_' + modelname + '_' + faultType + str(int(log_mag*100)) \
                   + '_EpCount_' + str(no_datasets) + '.ds'
    datasetFile = open(saveFilename, 'wb')  # Creating file to write
    pickle.dump(dataset_list, datasetFile)
    datasetFile.close()
