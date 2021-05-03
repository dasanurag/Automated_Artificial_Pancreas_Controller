import gym
import numpy as np

from stable_baselines.sac.policies import LnMlpPolicy, FeedForwardPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import SAC
from stable_baselines import ACKTR
from stable_baselines.common import set_global_seeds, make_vec_env
import gym
import simglucose
import argparse
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

from gym.envs.registration import register

def risk_event(BG_last_hour, **kwargs):
    if BG_last_hour[-1] > 180:
        return -1
    elif BG_last_hour[-1] < 70:
        return -2
    else:
        return 1

def reward_target(bg_hist, **kwargs):
    bg = bg_hist[-1]
    if bg < 70 or bg> 180:
        return -3 * np.abs(bg - 120)
    else:
        return -1 * np.abs(bg - 120)

def magni_reward(bg_hist, **kwargs):
    bg = max(1, bg_hist[-1])
    fBG = 3.5506*(np.log(bg)**.8353-3.7932)
    risk = 10 * (fBG)**2
    return -1*risk

def cameron_reward(bg_hist, **kwargs):
    bg = bg_hist[-1]
    a = .2370  # 1/(mg/dL)
    b = -36.21
    c = 6.0e-5  # (1/(mg/dL)**3)
    d = 177  # mg/dL
    if bg < d:
        risk = a*bg+b+(c*(d-bg)**3)
    else:
        risk = a*bg+b
    return -1*risk

def risk_diff(BG_last_hour, **kwargs):
    if len(BG_last_hour) < 2:
        return 0
    else:
        _, _, risk_current = risk_index([BG_last_hour[-1]], 1)
        _, _, risk_prev = risk_index([BG_last_hour[-2]], 1)
        return risk_prev - risk_current


def get_group(group):
    if ((group == 'adult') or (group == 'adolescent') or (group == 'child')):
        return group
    else:
        assert False, "Age Group must be valid ('adult' or 'adolescent' or 'child')"

def make_env(env_id, i, reward_fun, seed=0):
    def _init():
        patient_id = env_id.split('#')[0] + str(i)
        register(id='simglucose-' + patient_id + '-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': env_id, 'reward_fun': reward_fun})

        env = gym.make('simglucose-' + patient_id + '-v0')
        env.seed(seed)
        print(env_id)
        return env
    set_global_seeds(seed)
    return _init

def get_reward(r):
    if r == 'magni_reward':
        return magni_reward
    elif r == 'cameron_reward':
        return cameron_reward
    elif r == 'risk_event':
        return risk_event
    elif r == 'reward_target':
        return reward_target
    elif r == 'default':
        return risk_diff
    else:
        assert False, "Reward must be valid ('magni_reward', 'cameron_reward', 'risk_event', 'reward_target')"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--age_group", default="adult")
    parser.add_argument("-r", "--reward", default="default")
    args = parser.parse_args()
    group = get_group(args.age_group)
    reward_fun = get_reward(args.reward)

    env = DummyVecEnv([make_env(group + '#0{}'.format(str(i).zfill(2)), i, reward_fun) for i in range(1, 11)])
    model = ACKTR(MlpLstmPolicy, env, verbose=1)
    model.learn(total_timesteps=256000)
    model.save("ACKTR_MlpLSTM_" + group + "_def_reward")

