import pandas as pd
import numpy as np
import time
import random
import os
import subprocess
# fix for MACOSX GUI crash
import sys
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")  # use this backend to prevent macosX bug
import matplotlib.pyplot as plt
from tkinter import *


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


# class LocalSearch:

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

    def calculate_goal(self, job_df=None):

        if job_df is not None:
            __job_df = job_df
        else:
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
        __temp_job_df.iloc[swap[0], :], __temp_job_df.iloc[swap[1], :] = \
            self.job_df.iloc[swap[1], :], self.job_df.iloc[swap[0], :]

        return __temp_job_df


class MovePool:

    def __init__(self, job_df):
        self.n_jobs = job_df.shape[0]
        self.move_pool = []

    def generate_moves(self):

        for i in range(self.n_jobs - 1):
            for j in range(i+1, self.n_jobs):
                self.move_pool.append((i, j))

    def move_shuffle(self):
        random.shuffle(self.move_pool)


class Memory:

    def __init__(self):
        self.memory = []
        self.current_length = len(self.memory)

    def update_memory(self, item):
        self.memory.append(item)
        # could be done faster by writing += 1
        self.current_length += 1

    def remove_fifo(self):
        self.memory.pop(0)
        # could be done faster by writing -= 1
        self.current_length = len(self.memory)

    def clear_memory(self):

        self.memory = []
        self.current_length = 0


class Solution:

    def __init__(self, goalfunctionvalue=np.inf, move=(None, None), timestamp=np.inf):
        self.goalvalue = goalfunctionvalue
        self.move = move
        self.timestamp = timestamp

    def __str__(self):
        return "({0}, {1}, {2})".format(self.goalvalue, self.move, self.timestamp)


class SolutionMemory(Memory):

    def __init__(self, max_length=None):
        super().__init__()
        self.max_length = max_length
        self.best_solution = Solution(np.inf, (None, None))  # initialize by infinite goalfunction

    def update_memory(self, solution=Solution()):
        self.memory.append(solution)
        if solution.goalvalue < self.best_solution.goalvalue:
            self.best_solution = solution

        self.current_length += 1

    def last_solution(self):
        return self.memory[-1]

    def plot_memory(self, starttime):
        import matplotlib.pyplot as plt
        times = [solution.timestamp - starttime for solution in self.memory]
        values = [solution.goalvalue for solution in self.memory]
        plt.plot(times, values, label='solution space')
        plt.legend()
        plt.xlabel('runtime (s)')
        plt.ylabel('goalfunction: maximal lateness')
        plt.title('Local Search for minimal maximal lateness')

        filepath = 'img/fig' + str(starttime) + '.png'
        plt.savefig(filepath)
        os.chmod(filepath, 777)
        subprocess.Popen(['open ' + filepath], shell=True)


class TabuList(Memory):

    def __init__(self):
        super().__init__()


class LocalSearch:

    def __init__(self, jobschedule, max_iter, x_improvements, track_solution=True, EDD_rule=True):

        self.instance = jobschedule  # jobschedule object
        if EDD_rule:
            self.instance.EDDRule()
            self.instance.update_goal()
        self.pool = MovePool(self.instance.job_df)
        self.pool.generate_moves()

        # metaparameters
        self.max_iter = max_iter
        self.x_improvements = x_improvements
        self.track_solution = track_solution
        self.verbose = True

        # memory structures
        self.solution_memory = SolutionMemory()
        self.best_solution_memory = SolutionMemory()
        self.first_x_memory = SolutionMemory(max_length=self.x_improvements)

        # start clock
        self.starttime = time.time()
        self.best_solution_memory.update_memory(solution=Solution(goalfunctionvalue=self.instance.goalvalue,
                                                                  move=(None, None),
                                                                  timestamp=self.starttime))

    def solve(self, verbose=True):

        for i in range(self.max_iter):
            if verbose:
                print("Iteration ", i)
            local_optimum = True
            improvement_found = False
            iteration_memory = SolutionMemory()

            for swap in self.pool.move_pool:

                # Start search in local neighbourhood
                ts = time.time()
                goalval = self.instance.calculate_goal(self.instance.swap_jobs(swap=swap))
                iteration_memory.update_memory(Solution(goalfunctionvalue=goalval,
                                                        move=swap,
                                                        timestamp=ts))

                # Add swap evaluation to solution memory (could be inefficient)
                if self.track_solution:
                    self.solution_memory.update_memory(solution=Solution(goalfunctionvalue=goalval,
                                                                         move=swap,
                                                                         timestamp=ts))

                # checks if local neighbour is better than last encountered solution
                if goalval < self.best_solution_memory.last_solution().goalvalue:
                    self.first_x_memory.update_memory(solution=Solution(goalfunctionvalue=goalval,
                                                                        move=swap,
                                                                        timestamp=ts))
                    improvement_found = True
                    if verbose:
                        print("first x improving solutions:")
                        for sol in self.first_x_memory.memory:
                            print(sol)

                if self.first_x_memory.current_length >= self.first_x_memory.max_length:
                    final_solution = self.first_x_memory.best_solution
                    if verbose:
                        print("Max amount of improvements found")
                        print("Best Improving Solution: \n", final_solution)
                        print('breaking current local search...')

                    self.instance.update_job_df(self.instance.swap_jobs(swap=swap))
                    self.instance.update_goal()
                    self.best_solution_memory.update_memory(solution=final_solution)

                    # improvement found, no local optimum
                    local_optimum = False
                    self.first_x_memory.clear_memory()  # ready for next iteration
                    break

            if verbose:
                print('All swaps evaluated...')

            if improvement_found:
                # If less than x improving moves are found in local neighbourhood
                # Choose best solution from improving moves
                final_solution = self.first_x_memory.best_solution
                self.instance.update_job_df(self.instance.swap_jobs(final_solution.move))
                self.instance.update_goal()

                # update best solution memory
                self.best_solution_memory.update_memory(solution=final_solution)

                # improvement found, no local optimum
                local_optimum = False
                self.first_x_memory.clear_memory()  # ready for next iteration

            # case of local otpimum
            if local_optimum:
                if verbose:
                    print('No improving solutions are found, stuck in a local optimum')
                    print('Best Global Solution encounterd: {}'.format(self.best_solution_memory.best_solution))
                    print('Best Solution found in local neighbourhood: {}'.format(iteration_memory.best_solution))

                swap = iteration_memory.best_solution.move
                self.instance.update_job_df(self.instance.swap_jobs(swap=swap))
                self.instance.update_goal()

                self.best_solution_memory.update_memory(solution=iteration_memory.best_solution)
                self.first_x_memory.clear_memory()  # ready for next iteration

        if verbose:
            print("Total runtime: ", time.time() - self.starttime)

        self.best_solution_memory.plot_memory(self.starttime)


def main2():
    if True:
        # Initialization
        instance = JobSchedule('instance.csv')  # initiate job schedule object
        instance.EDDRule()  # initialize earliest due date
        instance.update_goal()
        pool = MovePool(instance.job_df)  # initiate move pool object
        pool.generate_moves()  # generate moves

        # initialize memory structures
        solution_memory = SolutionMemory()
        best_solution_memory = SolutionMemory()
        first_x_memory = SolutionMemory(max_length=15)
        track_solution_memory = True

        # basis for timestamp calculation
        starttime = time.time()
        best_solution_memory.update_memory(solution=Solution(instance.goalvalue, (None, None), starttime))

    for i in range(10):
        print('iteration ', i)
        local_optimum = True
        improvement_found = False
        iteration_memory = SolutionMemory()

        for swap in pool.move_pool:
            # start search in local neighbourhood
            ts = time.time()
            # iterate over all neighbours
            goalval = instance.calculate_goal(instance.swap_jobs(swap))
            iteration_memory.update_memory(Solution(goalval, swap, ts))

            # add the swap to the solution memory (inefficient)
            if track_solution_memory:
                solution_memory.update_memory(solution=Solution(goalval, swap, ts))

            # checks if local neighbour is better than last encountered solution
            if goalval < best_solution_memory.last_solution().goalvalue:
                first_x_memory.update_memory(solution=Solution(goalval, swap, ts))
                # best solution of solutions is updated
                print('first x improving solutions: ')
                for sol in first_x_memory.memory:
                    print(sol)
                improvement_found = True

                if first_x_memory.current_length >= first_x_memory.max_length:
                    print('max amount of improvements found')
                    final_solution = first_x_memory.best_solution
                    print('best solution found: \n', final_solution)
                    instance.update_job_df(instance.swap_jobs(swap))
                    instance.update_goal()

                    # update best solution memory
                    best_solution_memory.update_memory(solution=final_solution)

                    local_optimum = False
                    first_x_memory.clear_memory()  # ready for next iteration
                    print('breaking current local search...')
                    break  # breaks out local neighbourhood search

        print('all swaps evaluated...')
        if improvement_found:
            final_solution = first_x_memory.best_solution
            print('improving solution found: ', final_solution)
            instance.update_job_df(instance.swap_jobs(final_solution.move))
            instance.update_goal()

            # update best solution memory
            best_solution_memory.update_memory(solution=final_solution)

            local_optimum = False
            first_x_memory.clear_memory()  # ready for next iteration

        # When reaching end of local search
        if local_optimum:
            print('No improving solution found, stuck in local optimum...')
            print('Best solution found: {}'.format(solution_memory.best_solution.goalvalue))
            print('Beste lokale oplossing: {}'.format(iteration_memory.best_solution))

            swap = iteration_memory.best_solution.move
            instance.update_job_df(instance.swap_jobs(swap))
            instance.update_goal()

            best_solution_memory.update_memory(solution=iteration_memory.best_solution)
            first_x_memory.clear_memory()  # ready for next iteration

    print('Total runtime: ', time.time() - starttime)
    best_solution_memory.plot_memory(starttime)


def main():

    # initialize GUI starter
    guistarter = True
    if guistarter:
        root = Tk()
        metaparamdict = {"max_iter": 5, "x_improvements": 15}

        def LocalSearchGui(root, fieldnames):
            win = root
            entries = []
            for fieldname, fieldvalue in zip(fieldnames.keys(), fieldnames.values()):
                row = Frame(win)
                lab = Label(row, width=15, text=fieldname)
                ent = Entry(row)
                ent.insert(0, fieldvalue)
                row.pack(side=TOP, fill=X)
                lab.pack(side=LEFT)
                ent.pack(side=RIGHT)
                ent.focus()
                ent.bind('<Return>', (lambda event: win.quit()))
                entries.append(ent)
            return entries

        # LocalSearch Metaparam:
        entries = LocalSearchGui(root, metaparamdict)

        # LocalSearch Verbosity:
        verbose = BooleanVar()
        verbosebtn = Checkbutton(root, text='Verbose?', variable=verbose, onvalue=True, offvalue=False)
        verbosebtn.select()
        verbosebtn.pack()

        # Run button
        btn = Button(root, text='Run!', command=root.quit).pack()

        def fetch_entries(entries, fieldnames):

            for fieldname, entry in zip(fieldnames.keys(), entries):
                fieldnames.update({fieldname: int(entry.get())})
            return fieldnames

        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)
        root.mainloop()
        metaparameters = fetch_entries(entries, metaparamdict)

    instance = JobSchedule(pathname='instance.csv')
    algorithm = LocalSearch(instance, max_iter=metaparameters['max_iter'],
                            x_improvements=metaparameters['x_improvements'])

    algorithm.solve(verbose=verbose.get())


if __name__ == '__main__':
    main()
