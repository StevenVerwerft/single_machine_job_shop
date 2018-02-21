import pandas as pd
import numpy as np
import timeit


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
        __instance['tot_times'] = __instance['p_times'].values +\
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


instance = Instance2('instance.csv')
print(instance.head())

# import time

# start = time.clock()
# for i in range(1000):
#     instance1 = Instance('instance.csv')
# end = time.clock()
# print((end - start)/1000)