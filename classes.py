import pandas as pd
import numpy as np
import timeit
import random

class Instance:
    """
    An instance object is a pandas DataFrame like object, containing specific information for a
    single machine job scheduling problem. It has following data attributes:

    -job_id: identification number for a specific job in the job sequence
    -group_id: identification number for the product group a job belongs to
    -p_times: the processing times for each individual job on the machine
    -s_fracs: fractions of the processing time for each individual job to calculate the setup time
    -d_dates: due dates, time flags when each job should be processed by the machine

    """

    def __init__(self, pathname):
        self.pathname = pathname
        self.instance = self.read_instance()

        self.n_jobs = self.instance.shape[0]

        self.instance = self.calculate_group_processing_times()
        self.instance = self.calculate_waiting_time()
        self.instance = self.calculate_lateness()

        self.solution = self.calculate_goal_function()

    def read_instance(self):
        """
        Reads .csv file into the instance object
        """
        instance = pd.read_csv(self.pathname, sep=',')
        return instance

    def calculate_group_processing_times(self):
        """
        Checks if consequent jobs belong to the same job family. Which would reduce the job's setup time to none.
        """
        __instance = self.instance.copy()
        __g_sequence = np.ones(self.n_jobs)
        __g_sequence[1:] = np.where(__instance['group_id'].values[:-1] == __instance['group_id'].values[1:], 0, 1)
        __instance['tot_times'] = __instance['p_times'].values + \
                                  __instance['p_times'].values * __instance['s_fracs'].values * __g_sequence

        return __instance

    def calculate_waiting_time(self):
        """

        :return:
        """
        __instance = self.instance.copy()
        __W = np.zeros(self.n_jobs)
        __W[1:] = np.cumsum(__instance['tot_times'].values[:-1])
        __instance['w_time'] = __W

        return __instance

    def calculate_lateness(self):
        """

        :return:
        """
        __instance = self.instance.copy()
        __L = __instance['tot_times'] + __instance['w_time'] - __instance['d_dates']
        __instance['lateness'] = np.maximum(np.zeros(self.n_jobs), __L)

        return __instance

    def calculate_goal_function(self):
        """

        :return:
        """

        __solution = self.instance['lateness'].max()
        return __solution


class Instance2(Instance):

    def __init__(self):
        super().__init__(self)


class LocalSearch:

    def __init__(self, max_iter):
        self.max_iter = max_iter

    def solve(self, instance):
        for i in range(self.max_iter):
            print('Solving Iteration {}'.format(i))

    def EDD_rule(self, instance):
        return instance.sort_values(by='d_dates')


class Job:

    def __init__(self, id, family_id, process_time, setup_fraction, due_date):
        self.id = int(id)
        self.family_id = int(family_id)
        self.p_time = process_time
        self.s_frac = setup_fraction
        self.s_time = self.p_time * self.s_frac
        self.d_date = due_date

    def __str__(self):
        return "(id: {0}, family_id: {1}, p_time: {2}, " \
               "s_frac: {3}, s_time: {4}, d_date: {5})".format(self.id,
                                                               self.family_id,
                                                               round(self.p_time, 2),
                                                               round(self.s_frac, 2),
                                                               round(self.s_time, 2),
                                                               round(self.d_date, 2))


class JobSchedule:

    def __init__(self, pathname):
        """ Creates a pandas Dataframe object containing job information"""
        self.job_df = pd.read_csv(pathname, sep=',')
        self.job_df.columns = ['id', 'group_id', 'p_time', 's_frac', 'd_date']
        self.n_jobs = len(self.job_df)
        self.goalvalue = self.calculate_goal()

    def EDDRule(self):
        """Earliest Due Date Rule, sorts the jobschedule by earliest due date"""
        self.job_df = self.job_df.sort_values(by='d_date')

    def calculate_goal(self):

        __job_df = self.job_df.__deepcopy__()

        # 1. Calculate processing times for groups
        g_sequence = np.ones(self.n_jobs)  # Identifier for consequent group jobs

        # We check of consequent jobs belong to the same job family, which would reduce setup time
        # The first job in the sequence always needs a setup time!
        g_sequence[1:] = np.where(__job_df['group_id'].values[:-1] == __job_df['group_id'].values[1:], 0, 1)

        # 2. Calculate the total processing times for the current job sequenec
        __job_df['tot_time'] = __job_df['p_time'] + \
                               __job_df['s_frac'] * __job_df['p_time'] * g_sequence

        # 3. Calculate waiting time for each job, waiting time is cumulative total processing time before a job starts
        waiting_time = np.zeros(self.n_jobs)  # default waiting time is zero
        waiting_time[1:] = np.cumsum(__job_df['tot_time'].values[:-1])  # first job doesn't have a waiting time

        # 4. Calculate job lateness
        lateness = __job_df['tot_time'] + waiting_time - __job_df['d_date']
        __job_df['lateness'] = np.maximum(np.zeros(self.n_jobs), lateness)  # lateness cannot be negative

        # 5. calculate goalfunction (minimal maximal lateness)

        goalvalue = __job_df['lateness'].max()
        return goalvalue

    def update_goal(self):
        self.goalvalue = self.calculate_goal()

    def update_job_df(self, job_df):
        self.job_df = job_df

    def swap_jobs(self, swap):

        __temp_job_df = self.job_df.__deepcopy__()
        __temp_job_df.values[swap[0], :], __temp_job_df.values[swap[1], :] = self.job_df.values[swap[1], :]

        return __temp_job_df


class MovePool:

    def __init__(self, job_df):
        self.n_jobs = job_df.shape[0]
        self.move_pool = []

    def generate_moves(self):

        for i in range(self.n_jobs - 1):
            for j in range(i+1, self.n_jobs):
                self.move_pool.append((i, j))

    def shuffle_movepool(self):
        random.shuffle(self.move_pool)


class LocalSearch:

    def __init__(self, max_iter):
        self.max_iter = max_iter

    def evaluate_move(self, move, jobschedule, solution_memory):

        local_job_df = jobschedule.swap_jobs(swap=move)



def main():
    instance = JobSchedule('instance.csv')
    pool = MovePool(instance.job_df)
    pool.generate_moves()
    pool.shuffle_movepool()
    print(pool.move_pool)


if __name__ == '__main__':
    main()
