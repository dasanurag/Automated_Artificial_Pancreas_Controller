import gym

# Register gym environment. By specifying kwargs,
# you are able to choose which patient to simulate.
# patient_name must be 'adolescent#001' to 'adolescent#010',
# or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
from gym.envs.registration import register
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
# from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2, SAC
from simglucose.controller.base import Controller, Action
from simglucose.simulation.sim_engine import SimObj, batch_sim, sim
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario_gen import RandomScenario
# from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from datetime import timedelta
from datetime import datetime
from stable_baselines import ACKTR
from stable_baselines.common import set_global_seeds, make_vec_env

### Reward Functions ###
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

person_options = (['child#0{}'.format(str(i).zfill(2)) for i in range(1, 11)]+['adolescent#0{}'.format(str(i).zfill(2)) for i in range(1, 11)]+['adult#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])
for i,p in enumerate(person_options):

    patient_id = p.split('#')[0] + str(i + 1)
    # Create a simulation environment
    print(p)
    patient = T1DPatient.withName(p)
    register(id='simglucose-'+p+'-v0',entry_point='simglucose.envs:T1DSimEnv',kwargs={'patient_name': p},'reward_fun': reward_target)
    env = gym.make('simglucose-'+p+'-v0')

    model = ACKTR(MlpLstmPolicy, env, verbose=1)
    model.learn(total_timesteps=250000)
    model.save('mlplstm_trained-'+p+'-reward_target')
    print('Model Trained and Saved for : '+ p)
