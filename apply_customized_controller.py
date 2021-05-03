# from simglucose.simulation.user_interface import simulate
from simglucose.controller.base import Controller, Action
from simglucose.simulation.sim_engine import SimObj, batch_sim, sim
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.analysis.risk import risk_index
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
import gym
from simglucose.analysis.report import report
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.pid_ctrller import PIDController
from gym.envs.registration import register
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env, set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2
from stable_baselines import ACKTR
from datetime import timedelta
from datetime import datetime
import pandas as pd
import numpy as np
import os
import argparse

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

def make_env(env_id, i, seed=0):
    def _init():
        patient_id = env_id.split('#')[0] + str(i)
        register(id='simglucose-' + patient_id + '-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': env_id})

        env = gym.make('simglucose-' + patient_id + '-v0')
        env.seed(seed)
        print(env_id)
        return env
    set_global_seeds(seed)
    return _init

def get_group(group):
    if ((group == 'adult') or (group == 'adolescent') or (group == 'child')):
        return group
    else:
        assert False, "Age Group must be valid ('adult' or 'adolescent' or 'child')"

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
        assert False, "Reward must be valid ('magni_reward', 'cameron_reward', 'risk_event', 'reward_target', 'default')"

path = './results/'

os.makedirs(path, exist_ok = True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--age_group", default="adult")
    parser.add_argument("-r", "--reward", default="default")
    args = parser.parse_args()
    group = get_group(args.age_group)
    reward_fun = get_reward(args.reward)
    env_id = []
    person_options = ([group+'#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])

    for i,p in enumerate(person_options):
        patient_id = p.split('#')[0] + str(i + 1)
        # Create a simulation environment
        print(p)
        patient = T1DPatient.withName(p)
        register(id='simglucose-'+patient_id+'-v0',entry_point='simglucose.envs:T1DSimEnv',kwargs={'patient_name': p, 'reward_fun': reward_fun})
        env_id.append('simglucose-'+patient_id+'-v0')

    state = np.zeros((10,1))

    env0 = gym.make(env_id[0])
    env0 = DummyVecEnv([lambda: env0])
    state[0] = env0.reset()
    env1 = gym.make(env_id[1])
    env1 = DummyVecEnv([lambda: env1])
    state[1] = env1.reset()
    env2 = gym.make(env_id[2])
    env2 = DummyVecEnv([lambda: env2])
    state[2] = env2.reset()
    env3 = gym.make(env_id[3])
    env3 = DummyVecEnv([lambda: env3])
    state[3] = env3.reset()
    env4 = gym.make(env_id[4])
    env4 = DummyVecEnv([lambda: env4])
    state[4] = env4.reset()
    env5 = gym.make(env_id[5])
    env5 = DummyVecEnv([lambda: env5])
    state[5] = env5.reset()
    env6 = gym.make(env_id[6])
    env6 = DummyVecEnv([lambda: env6])
    state[6] = env6.reset()
    env7 = gym.make(env_id[7])
    env7 = DummyVecEnv([lambda: env7])
    state[7] = env7.reset()
    env8 = gym.make(env_id[8])
    env8 = DummyVecEnv([lambda: env8])
    state[8] = env8.reset()
    env9 = gym.make(env_id[9])
    env9 = DummyVecEnv([lambda: env9])
    state[9] = env9.reset()

    print('Environment Created')
    model_name = 'ACKTR_MlpLSTM_'+group+'_'+args.reward
    MODEL_PATH='Saved_models'
    tr_model = ACKTR.load(MODEL_PATH + '/' + model_name)

    t = 480                         ## number of time steps to evaluate. t = 480 is 1 day
    all_state = np.zeros((10,t))

    print('Simulation Started ... ...')
    for i in range(t):
        aa, _ = tr_model.predict(state)
        # print(aa)
        action = Action(basal=aa[0]/ 6000, bolus = 0)
        state[0], reward, done, _ = env0.step(action)
        action = Action(basal=aa[1]/ 6000, bolus = 0)
        state[1], reward, done, _ = env1.step(action)
        action = Action(basal=aa[2]/ 6000, bolus = 0)
        state[2], reward, done, _ = env2.step(action)
        action = Action(basal=aa[3]/ 6000, bolus = 0)
        state[3], reward, done, _ = env3.step(action)
        action = Action(basal=aa[4]/ 6000, bolus = 0)
        state[4], reward, done, _ = env4.step(action)
        action = Action(basal=aa[5]/ 6000, bolus = 0)
        state[5], reward, done, _ = env5.step(action)
        action = Action(basal=aa[6]/ 6000, bolus = 0)
        state[6], reward, done, _ = env6.step(action)
        action = Action(basal=aa[7]/ 6000, bolus = 0)
        state[7], reward, done, _ = env7.step(action)
        action = Action(basal=aa[8]/ 6000, bolus = 0)
        state[8], reward, done, _ = env8.step(action)
        action = Action(basal=aa[9]/ 6000, bolus = 0)
        state[9], reward, done, _ = env9.step(action)

        all_state[:,i] = state[:,0]

    print('Simulation Completed')
    result = np.zeros(10)
    for j in range(10):
        result[j] = len([I for I in all_state[j,:] if 70<=I<=180])/all_state.shape[1]
    print(result)
    # df = pd.DataFrame(all_state)
    # result = len([I for I in all_state if 70<=I<=180])/len(all_state)
    # print(result)
    # final_result = pd.DataFrame(list(zip(person_options, percent_time)), columns=['Paient_ID','result'])
    pd.DataFrame(all_state).to_csv(path+group+'_'+args.reward+'_'+'BG_response.csv')
    pd.DataFrame(result).to_csv(path+group+'_'+args.reward+'_'+'final_result.csv')
    print('Results Saved')
