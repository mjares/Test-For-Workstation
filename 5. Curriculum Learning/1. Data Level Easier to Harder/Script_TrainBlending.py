import os
import sys
from pathlib import Path

from stable_baselines3 import PPO
# from tensorflow.keras.callbacks import TensorBoard
# Adding local libs to path
parentFolder = Path(os.getcwd()).parent
sys.path.append(str(parentFolder) + '\\Z. Local mods')

from QuadcopterBlendEnv import QuadcopterBlendEnv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initial Conditions
goal_rew = 2000
match_rew = 0
switch_logic = 'Free'

# Easy
log_model_every = 1000000
timesteps = 2000000
load_model_path = None
operating_modes = {'Rotor': [0.01, 0.10],
                   'AttNoise': [0.01, 0.4]}

# Tensorboard
# tensorboard = TensorBoard(log_dir="logs\\{}".format('my_saved_model'))

reward_params = {'outofbounds': 1,
                 'finalgoal': goal_rew,
                 'timepenalty': 0.1,
                 'withinbounds': 0,
                 'matchenv': match_rew,
                 'bumpless': 0}
env = QuadcopterBlendEnv(action_space_type="ContinuousRestricted", switching_logic=switch_logic, verbose=True,
                         reward_params=reward_params, operating_modes=operating_modes)
obs = env.reset()
if load_model_path is None:
    model = PPO("MlpPolicy", env, verbose=1)
else:
    model = PPO.load(load_model_path, env)

t_step_count = 0
while t_step_count < timesteps:

    t_step_count = t_step_count + log_model_every
    model.learn(total_timesteps=log_model_every,  log_interval=10000)
    save_model_path = f"Agents/QuadBlendingAgent_MatchPenalty_{int(match_rew*10)}_Goal_{int(goal_rew)}" \
                      f"_Easy_{t_step_count}"
    model.save(save_model_path)
    print(f'Training in progress: {t_step_count}/{timesteps}')

print("Training complete - agent saved")

# Intermediate
log_model_every = 1000000
timesteps = 2000000
load_model_path = save_model_path
operating_modes = {'Rotor': [0.01, 0.20],
                   'AttNoise': [0.01, 0.8]}
reward_params = {'outofbounds': 1,
                 'finalgoal': goal_rew,
                 'timepenalty': 0.1,
                 'withinbounds': 0,
                 'matchenv': match_rew,
                 'bumpless': 0}
env = QuadcopterBlendEnv(action_space_type="ContinuousRestricted", switching_logic=switch_logic, verbose=True,
                         reward_params=reward_params, operating_modes=operating_modes)
obs = env.reset()
if load_model_path is None:
    model = PPO("MlpPolicy", env, verbose=1)
else:
    model = PPO.load(load_model_path, env)

t_step_count = 0
while t_step_count < timesteps:

    t_step_count = t_step_count + log_model_every
    model.learn(total_timesteps=log_model_every,  log_interval=10000)
    save_model_path = f"Agents/QuadBlendingAgent_MatchPenalty_{int(match_rew*10)}_Goal_{int(goal_rew)}" \
                      f"_Intermediate_{t_step_count}"
    model.save(save_model_path)
    print(f'Training in progress: {t_step_count}/{timesteps}')

print("Training complete - agent saved")

# Hard
log_model_every = 1000000
timesteps = 2000000
load_model_path = save_model_path
operating_modes = {'Rotor': [0.10, 0.30],
                   'AttNoise': [0.4, 1.2]}

# Tensorboard
# tensorboard = TensorBoard(log_dir="logs\\{}".format('my_saved_model'))

reward_params = {'outofbounds': 1,
                 'finalgoal': goal_rew,
                 'timepenalty': 0.1,
                 'withinbounds': 0,
                 'matchenv': match_rew,
                 'bumpless': 0}
env = QuadcopterBlendEnv(action_space_type="ContinuousRestricted", switching_logic=switch_logic, verbose=True,
                         reward_params=reward_params, operating_modes=operating_modes)
obs = env.reset()
if load_model_path is None:
    model = PPO("MlpPolicy", env, verbose=1)
else:
    model = PPO.load(load_model_path, env)

t_step_count = 0
while t_step_count < timesteps:

    t_step_count = t_step_count + log_model_every
    model.learn(total_timesteps=log_model_every,  log_interval=10000)
    save_model_path = f"Agents/QuadBlendingAgent_MatchPenalty_{int(match_rew*10)}_Goal_{int(goal_rew)}" \
                      f"_Hard_{t_step_count}"
    model.save(save_model_path)
    print(f'Training in progress: {t_step_count}/{timesteps}')

print("Training complete - agent saved")