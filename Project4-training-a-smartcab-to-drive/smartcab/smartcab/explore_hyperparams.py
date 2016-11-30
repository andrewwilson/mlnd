from agent import LearningAgent
from environment import Environment
from simulator import Simulator
from twiddle import twiddle
from multiprocessing import Pool

import datetime as dt
import pandas as pd
import numpy as np

def calc_error_score(sim):

    data = pd.read_csv(sim.log_file.name)
    data['average_reward'] = (data['net_reward'] / (data['initial_deadline'] - data['final_deadline']))
    data['reliability_rate'] = (data['success'] * 100)
    test_data = data[data['testing'] == True]
    avg_reward = test_data['average_reward'].mean()
    avg_reliability = test_data['reliability_rate'].mean()

    score = avg_reward * avg_reliability
    print "## score, reward, reliability", score, avg_reward, avg_reliability

    # twiddle attempts to minimise score, which it interprets as an error, so return -1* our score here.
    return -1 * score

def run_test(params):

    alpha = params[0]
    epsilon = params[1]
    epsilon_decay_rate = params[2]
    log_suffix = params[3]


    print "Running test with: alpha:", alpha, "epsilon", epsilon, "epsilon_decay_rate", epsilon_decay_rate, "log_suffix",log_suffix

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(
        verbose=False,
        text_output=False
    )

    # Create the driving agent
    agent = env.create_agent(LearningAgent,
                             learning=True,
                             alpha=alpha,
                             epsilon=epsilon,
                             epsilon_decay_rate=epsilon_decay_rate
                             #epsilon_decay_step=epsilon_decay_step
                             )

    # Follow the driving agent
    env.set_primary_agent(agent,
                          enforce_deadline=True)

    # Create the simulation
    sim = Simulator(env,
                    display=False,
                    update_delay=0,
                    log_metrics=True,
                    optimized=True,
                    text_output=False,
                    log_suffix=log_suffix)

    # Run the simulator
    sim.run(
        n_test=20,
        tolerance=0.01,
        max_train=500
    )

    # calculate error score
    score = calc_error_score(sim)
    return score


def run_test_wrapper(params):


    N = 20 # train N models with these params and average the scores after testing each of them.
    start = dt.datetime.now()

    task_args = [ params + [ "_" + str(i+1)] for i in range(N) ]

    print "Running ",N," tests with params",params

    scores = pool.map(run_test, task_args, chunksize=2)
    print "## Scores were:", scores

    score = np.mean(scores)
    print "## mean,std", score, np.std(scores)
    end = dt.datetime.now()
    print "## Test took: ", end - start
    return score


if __name__ == '__main__':


    # parameters to tune

    params = [
        0.5,    # alpha
        1.0,    # epsilon
        0.98,    # epsilon_decay_rate
    ]

    limits = [
        [0.001, 1.0], # alpha
        [0.01, 1.0],   # epsilon
        [0.001, 0.999],   # epsilon_decay_rate
    ]

    # initial changes for parameter exploration
    param_deltas = [
        0.1,
        0.1,
        0.01
    ]

    pool = Pool(processes=10)

    results = twiddle(params, param_deltas, run_test_wrapper, threshold=0.001, max_iter=50, scaling=2.0, limits=limits)
    print "Result:", results

