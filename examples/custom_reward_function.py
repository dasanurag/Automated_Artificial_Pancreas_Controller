import gym
from gym.envs.registration import register
import numpy as np


def risk_event(BG_last_hour):
    if BG_last_hour[-1] > 180:
        return -1
    elif BG_last_hour[-1] < 70:
        return -2
    else:
        return 1

def reward_target(bg_hist, **kwargs):
    bg = bg_hist[-1]
    if bg < 70 or bg> 180:
        return -1 * np.abs(bg - 120)
    else:
        return -3 * np.abs(bg - 120)

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


register(
    id='simglucose-adolescent2-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adult#001',
            'reward_fun': magni_reward}
)

env = gym.make('simglucose-adolescent2-v0')

reward = 1
done = False

observation = env.reset()
for t in range(150):
    env.render(mode='human')
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(observation)
    print("Reward = {}".format(reward))
    # if done:
    #     print("Episode finished after {} timesteps".format(t + 1))
    #     break
