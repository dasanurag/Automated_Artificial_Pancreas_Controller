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

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

from gym.envs.registration import register

register(
    id='simglucose-patient-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#002'}
)
env1 = gym.make('simglucose-patient-v0')

# model = SAC(LnMlpPolicy, env1, verbose=1)


# del model # remove to demonstrate saving and loading
# person_options = (['child#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
#                   ['adolescent#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
#                   ['adult#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])

#### train model on children
# child_options = (['child#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])

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

if __name__ == "__main__":
    env = DummyVecEnv([make_env('child#0{}'.format(str(i).zfill(2)), i) for i in range(1, 11)])
    # model = SAC(LnMlpPolicy, env, verbose=1)
    model = ACKTR(MlpLstmPolicy, env, verbose=1)
    model.learn(total_timesteps=256000)
    model.save("ACKTR_MlpLSTM_child_def_reward")

# for i,p in enumerate(child_options):
#     patient_id = p.split('#')[0] + str(i + 1)
    
#     register(
#         id='simglucose-' + patient_id + '-v0',
#         entry_point='simglucose.envs:T1DSimEnv',
#         kwargs={'patient_name': p}
#     )
    

#     env = gym.make('simglucose-' + patient_id + '-v0')
#     model = SAC(LnMlpPolicy, env, verbose=1)
#     print(p, patient_id)
#     model.learn(total_timesteps=250000)
#     print("Finished training for " + patient_id)
#     if i == 5:
#         model.save("SAC_LnMlp_child_def_reward")        
# model.save("SAC_LnMlp_child_def_reward")



# #### train model on adolescents
# model = SAC(LnMlpPolicy, env1, verbose=1)

# adolescent_options = (['adolescent#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])

# for i,p in enumerate(adolescent_options):
#     patient_id = p.split('#')[0] + str(i + 1)
    
#     register(
#         id='simglucose-' + patient_id + '-v0',
#         entry_point='simglucose.envs:T1DSimEnv',
#         kwargs={'patient_name': p}
#     )
    
#     env = gym.make('simglucose-' + patient_id + '-v0')
#     print(p, patient_id)
#     model.learn(total_timesteps=250000)
#     print("Finished training for " + patient_id)
#     if i == 5:
#         model.save("SAC_LnMlp_adolescent_def_reward")
# model.save("SAC_LnMlp_adolescent_def_reward")


# #### train model on adolescents
# model = SAC(LnMlpPolicy, env1, verbose=1)

# adult_options = (['adult#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])

# for i,p in enumerate(adult_options):
#     patient_id = p.split('#')[0] + str(i + 1)
    
#     register(
#         id='simglucose-' + patient_id + '-v0',
#         entry_point='simglucose.envs:T1DSimEnv',
#         kwargs={'patient_name': p}
#     )
    
#     env = gym.make('simglucose-' + patient_id + '-v0')
#     print(p, patient_id)
#     model.learn(total_timesteps=250000)
#     print("Finished training for " + patient_id)
#     if i == 5:
#         model.save("SAC_LnMlp_adult_def_reward")        

# model.save("SAC_LnMlp_adult_def_reward")





