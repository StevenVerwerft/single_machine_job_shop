import numpy as np
import pandas as pd

pd.set_option('display.width', 140)


def read_instance(name):

    assert type(name) == str

    _instance = pd.read_csv(name, sep=',')
    _instance['s_times'] = _instance.s_fracs * _instance.p_times
    _instance = _instance[["job_id", "group_id", "p_times", "s_times", "d_dates"]]

    return _instance


def EDD_rule(_job_df):
    """
    Earliest due date rule, this is a simple heuristic that can be used to intialise the local search
    """
    _job_df = _job_df.copy()

    return _job_df.sort_values(by='d_dates')


def calculate_group_processing_times(_job_df):
    """
    Checks if consequent jobs belong to the same job family, which would reduce the job's setup time to none.
    """
    n = _job_df.shape[0]

    """
    The _g_sequence is an array with identifiers whether consequent jobs belong to the same job family.
    In these cases the job setup time should be discarded (multiply by 0)
    
    total times are then calculating by adding job process times and setup times together.
    """

    _job_df = _job_df.copy()  # skip reference to global object
    # Evaluate consequent groups
    _g_sequence = np.ones(n)
    _g_sequence[1:] = np.where(_job_df.group_id.values[:-1] == _job_df.group_id.values[1:], 0, 1)
    _job_df['g_sequence'] = _g_sequence
    _job_df['s_times'] = _job_df['s_times'] * _job_df['g_sequence']
    _job_df['tot_time'] = _job_df['p_times'] + _job_df['s_times']

    return _job_df


def calculate_waiting_time(_job_df):

    """
    The input is a dataframe structure containing the current job sequence and the relevant times for each job
    (processing time, setup time and due date)

    This function calculates for the given job schedule the waiting time in the queue for each job.
    """

    n = _job_df.shape[0]

    """
    The waiting time is an array which indicates the amount of time a job has been waiting before being scheduled
    on the machine. 
    """

    _job_df = _job_df.copy()  # skip refernce to global object

    # Calculate waiting times
    _W = np.zeros(n)
    _W[1:] = np.cumsum(_job_df['tot_time'].values[:-1])  # First job doesn't have a waiting time.
    _job_df['w_time'] = _W

    return _job_df


def calculate_lateness(_job_df):

    """
    The job's lateness is the time between the finishing of a job and its due date.
    """

    n = _job_df.shape[0]
    _job_df = _job_df.copy()

    _L = _job_df['tot_time'] + _job_df['w_time'] - _job_df['d_dates']
    _job_df['lateness'] = np.maximum(np.zeros(n), _L)

    return _job_df


def calculate_goal_function(_job_df):
    """
    Evaluates job's latenesses for a given job schedule. The maximal value for job's lateness in the lateness array is
    considered the goal function to minimize.
    """

    _goalf = _job_df['lateness'].max()

    return _goalf


def evaluate_job_sequence(_job_df):
    """
    Evaluates group family consequences
    Calculates total processing times
    Calculates waiting times
    Calculates job lateness
    Calculates maximal job lateness
    """

    _job_df = _job_df.copy()

    _groups = calculate_group_processing_times(_job_df)
    _waiting = calculate_waiting_time(_groups)
    _lateness = calculate_lateness(_waiting)
    _goalf = calculate_goal_function(_lateness)
    return _goalf


def create_swaps(_job_df):

    """
    Creates a list of value pairs representing the different job combinations which could be swapped
    """
    from itertools import combinations

    n = _job_df.shape[0]
    _swaps = [(i, j) for i, j in combinations(range(n), 2)]

    return _swaps


def perform_swap(_job_df, _swap):

    _temp_job_df = _job_df.copy()
    _temp_job_df.iloc[_swap[0]], _temp_job_df.iloc[_swap[1]] = _job_df.iloc[_swap[1]], _job_df.iloc[_swap[0]]

    return _temp_job_df

