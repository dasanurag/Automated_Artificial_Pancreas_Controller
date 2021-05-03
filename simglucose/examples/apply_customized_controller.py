# from simglucose.simulation.user_interface import simulate
from simglucose.controller.base import Controller, Action
from simglucose.simulation.sim_engine import SimObj, batch_sim, sim
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario_gen import RandomScenario
# from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
import gym
from simglucose.analysis.report import report
from simglucose.controller.basal_bolus_ctrller import BBController
# Register gym environment. By specifying kwargs,
# you are able to choose which patient to simulate.
# patient_name must be 'adolescent#001' to 'adolescent#010',
# or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
from gym.envs.registration import register
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from datetime import timedelta
from datetime import datetime
import pandas as pd

class MyController(Controller):
    def __init__(self, curr_state, model):
        # self.init_state = init_state
        self.state = curr_state
        self.model = model

    def policy(self, observation, reward, done, **info):
        '''
        Every controller must have this implementation!
        ----
        Inputs:
        observation - a namedtuple defined in simglucose.simulation.env. For
                      now, it only has one entry: blood glucose level measured
                      by CGM sensor.
        reward      - current reward returned by environment
        done        - True, game over. False, game continues
        info        - additional information as key word arguments,
                      simglucose.simulation.env.T1DSimEnv returns patient_name
                      and sample_time
        ----
        Output:
        action - a namedtuple defined at the beginning of this file. The
                 controller action contains two entries: basal, bolus
        '''
        # sample_time = kwargs.get('sample_time')
        aa =  self.model.predict(observation)

        # action = Action(basal=aa, bolus=0)
        # print(type(aa),aa[0])
        return Action(basal=aa[0] / 6000, bolus=0)

    def reset(self):
        '''
        Reset the controller state to inital state, must be implemented
        '''
        self.state = self.init_state


def magni_reward(bg_hist, **kwargs):
    bg = max(1, bg_hist[-1])
    fBG = 3.5506*(np.log(bg)**.8353-3.7932)
    risk = 10 * (fBG)**2
    return -1*risk
# register(
#     id='simglucose-adolescent1-v0',
#     entry_point='simglucose.envs:T1DSimEnv',
#     kwargs={'patient_name': 'adolescent#001'}
# )
#
# env = gym.make('simglucose-adolescent1-v0')
import os

path = './results/SIM_BB_magni'
try:
    os.makedirs(path)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

percent_time = []
now = datetime.now()
start_time = datetime.combine(now.date(), datetime.min.time())
person_options = (['child#0{}'.format(str(i).zfill(2)) for i in range(1, 11)]+['adolescent#0{}'.format(str(i).zfill(2)) for i in range(1, 11)]+['adult#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])
for i,p in enumerate(person_options):

    # patient_id = p.split('#')[0] + str(i + 1)
    # Create a simulation environment
    print(p)
    patient = T1DPatient.withName(p)
    sensor = CGMSensor.withName('Dexcom', seed=1)
    pump = InsulinPump.withName('Insulet')
    scenario = RandomScenario(start_time=start_time, seed=1)
    env = T1DSimEnv(patient, sensor, pump, scenario)
    # tr_model = PPO2.load("sim-adl-02")
    # in_state = env.reset()
    # ctrller = MyController(0, model = tr_model) #BBController() #
    controller = BBController()
    sim_i = [SimObj(env, controller, timedelta(days=1), animate=False, path=path)]

    # sim_instances.simulate()
    # sim_instances.save_results()
    # result =  sim(sim_instances[0])
    # print('Simulation Completed!')
    # # df = pd.concat(result, keys=[s.env.patient.name for s in sim_instances])
    # results, ri_per_hour, zone_stats, figs, axes = report(path, df=result)
    results1 = sim(sim_i[0])
    # print(results1)
    pd_result= pd.DataFrame(results1)
    bg = list(pd_result['BG'])
    print(type(bg), bg[0], type(bg[0]))

    percent_time.append((len([I for I in bg if 70<=I<=180])/len(bg))*100)

final_result = pd.DataFrame(list(zip(person_options, percent_time)), columns=['Paient_ID','Percent_Time'])
final_result.to_csv(path+'/Final Result.csv', index=False)







# simulate(controller=ctrller)
