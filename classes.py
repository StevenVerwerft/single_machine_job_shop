
# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------------------------------------------
# CLASSES
# ----------------------------------------------------------------------------------------------------------------------


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

    def create_solution_df(self, starttime):
        """
        The Solution Memories' memory is transformed to a pandas DataFrame object, which can be manipulated or
        saved.
        """

        goalvalues = [solution.goalvalue for solution in self.memory]
        timestamps = [solution.timestamp - starttime for solution in self.memory]

        solution_df = pd.DataFrame(data={
            "goalvalues": goalvalues,
            "timestamps": timestamps
        })

        return solution_df


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
            max_improvements_reached = False
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
                    if verbose and False:
                        print("first x improving solutions:")
                        for sol in self.first_x_memory.memory:
                            print(sol)

                # when max amount x, of improvements is found, break the local neighbourhood search
                if self.first_x_memory.current_length >= self.first_x_memory.max_length:
                    final_solution = self.first_x_memory.best_solution
                    if verbose and False:
                        print("Max amount of improvements found")
                        print("Best Improving Solution: \n", final_solution)
                        print('breaking current local search...')

                    self.instance.update_job_df(self.instance.swap_jobs(swap=swap))
                    self.instance.update_goal()
                    self.best_solution_memory.update_memory(solution=final_solution)
                    if verbose:
                        print('added solution {} to best solution memory'.format(final_solution))
                    # improvement found, no local optimum
                    local_optimum = False
                    max_improvements_reached = True
                    self.first_x_memory.clear_memory()  # ready for next iteration
                    break

            if verbose and False:
                print('All swaps evaluated...')

            if improvement_found and not max_improvements_reached:
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
                if verbose and False:
                    print('No improving solutions are found, stuck in a local optimum')
                    print('Best Global Solution encounterd: {}'.format(self.best_solution_memory.best_solution))
                    print('Best Solution found in local neighbourhood: {}'.format(iteration_memory.best_solution))

                swap = iteration_memory.best_solution.move
                self.instance.update_job_df(self.instance.swap_jobs(swap=swap))
                self.instance.update_goal()

                self.best_solution_memory.update_memory(solution=iteration_memory.best_solution)
                if verbose:
                    print('Added Non-improving solution {} to best solution memory')

                self.first_x_memory.clear_memory()  # ready for next iteration

        if verbose:
            print("Total runtime: ", time.time() - self.starttime)

        self.best_solution_memory.plot_memory(self.starttime)


class LocalSearchGui(Frame):

    def __init__(self, parent, fieldnames, **options):

        self.fieldnames = fieldnames
        Frame.__init__(self, parent, **options)
        self.pack()
        self.entries = []

        for (fieldname, fieldvalue) in fieldnames.items():

            row = Frame(self)
            lab = Label(row, width=15, text=fieldname)
            ent = Entry(row)
            ent.insert(0, fieldvalue)
            row.pack(side=TOP, fill=X)
            lab.pack(side=LEFT)
            ent.pack(side=RIGHT)
            ent.focus()
            ent.bind('<Return>', (lambda event: self.quit))
            self.entries.append(ent)

        self.verbose = BooleanVar()
        verbosebtn = Checkbutton(self, text='Verbose?', variable=self.verbose,
                                 onvalue=True, offvalue=False)
        verbosebtn.select()
        verbosebtn.pack()
        runbtn = Button(self, text='Run!', command=parent.quit).pack()

    def fetch_entries(self):

        for (fieldname, entry) in zip(self.fieldnames.keys(), self.entries):
            self.fieldnames.update({fieldname: int(entry.get())})
        return self.fieldnames
