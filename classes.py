
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

    def plot_memory(self, starttime, label=None, **kwargs):
        import matplotlib.pyplot as plt
        times = [solution.timestamp - starttime for solution in self.memory]
        values = [solution.goalvalue for solution in self.memory]

        if label is not None:
            plt.plot(times, values, label='{}'.format(label))
        else:
            plt.plot(times, values, label='Solution Path')
        plt.legend()
        plt.xlabel('runtime (s)')
        plt.ylabel('goalfunction: maximal lateness')
        plt.title('Local Search for minimal maximal lateness')

        filepath = 'img/fig' + str(starttime) + '.png'

        try:
            filepath = str(kwargs['custom_name']) + '.png'
        except:
            pass
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

    def __init__(self, max_length=None, remove='fifo'):
        super().__init__()
        self.max_length = max_length
        self.remove = remove
        self.frequency_memory = []
        self.tabu_memory = []

    def update_tabulist(self, move):

        if self.max_length is not None:
            if len(self.tabu_memory) >= self.max_length:
                self.remove_move()
        self.tabu_memory.append(move)

    def remove_move_fifo(self):
        try:
            self.tabu_memory.pop(0)
        except IndexError:
            if len(self.tabu_memory) == 0:
                print('Tabu Memory already empty!')
            else:
                pass

    def remove_move_random(self):

        print('move to be deleted: {}'.format(random.choice(self.tabu_memory)))
        print('random removal to be implemented...')

    def remove_move(self):

        if self.remove == 'fifo':
            self.remove_move_fifo()
        elif self.remove == 'random':
            self.remove_move_random()

    def check_tabu_status(self, move):
        """checks if the move passed to the tabulist object is tabu on the active memory"""

        if move in self.tabu_memory:
            # print('the move is in the active tabulist')
            # print('find another move!')
            return True  # The move cannot be accepted, check for another move!

        else:
            return False  # The move can be accepted!

    def clear_tabulist(self):

        self.tabu_memory = []

    def update_frequency_memory(self, move):

        move_found = False
        for tabumove in self.frequency_memory:
            if tabumove.move == move:
                tabumove.update_frequency()
                # print('updated frequency: ', tabumove)
                move_found = True
                break  # escape from the search

        # if move not found in frequencymemory
        # happens when it's the first time a move has been selected
        if not move_found:
            self.frequency_memory.append(TabuMove(move, frequency=1))


class TabuMove:

    def __init__(self, move, frequency):
        self.move = move
        self.frequency = frequency

    def update_frequency(self):
        self.frequency += 1

    def __str__(self):
        return 'Move(({}, {}), {})'.format(self.move[0], self.move[1], self.frequency)

    def __repr__(self):
        return str(self)


class LocalSearch:

    def __init__(self, jobschedule, max_iter, x_improvements, track_solution=True, EDD_rule=True,
                 tabulength=None, tabu_remove='fifo'):

        """

        :param jobschedule: Jobschedule object
        :param max_iter: number of iterations for the local search algorithm
        :param x_improvements: number of local improving solutions should be found before moving to next iteration
        :param track_solution: use a memory to track the goalvalue of every evauated move (inefficient)
        :param EDD_rule: use the Earliest Due Date rule to initiate a good solution
        :param tabulength: the maximal amount of moves on the tabulist
        :param tabu_remove: the method to remove moves from the tabulist if max tabulength reached
        (options: 'fifo', 'random')
        """

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
        self.tabu_length = tabulength
        self.tabu_remove = tabu_remove

        # memory structures
        self.solution_memory = SolutionMemory()
        self.best_solution_memory = SolutionMemory()
        self.first_x_memory = SolutionMemory(max_length=self.x_improvements)
        self.tabu_list = TabuList(max_length=self.tabu_length, remove=self.tabu_remove)

        if self.tabu_length is not None:
            self.use_tabu_memory = True
        else:
            self.use_tabu_memory = False

        # start clock
        self.starttime = time.time()
        self.best_solution_memory.update_memory(solution=Solution(goalfunctionvalue=self.instance.goalvalue,
                                                                  move=(None, None),
                                                                  timestamp=self.starttime))

    def solve(self, verbose=True, **kwargs):

        try:
            show_iter = kwargs['show_iter']
        except KeyError:
            show_iter = True

        for i in range(self.max_iter):
            if verbose or show_iter:
                print(50*'--')
                print("Iteration ", i)
                print(50*'--')
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
                # also checks of this neighbour is a non-tabu solution, tabu solutions will never be put in
                # the first-x memory structure.
                if goalval < self.best_solution_memory.last_solution().goalvalue:
                    print('current local solution: ', Solution(goalval, swap, ts))
                    print('last best solution: ', self.best_solution_memory.last_solution())
                    if self.use_tabu_memory \
                            and self.tabu_list.check_tabu_status(swap):
                        print('I\'m tabu', swap)
                        print('tabulist: ', self.tabu_list.tabu_memory)
                        # if statement true, the selected move is tabu and cannot be set as the final solution
                        # the flow will be directed to the evaluation of the next swap

                        # if the statement is false, the tabulist will update its active memory and its
                        # frequency memory with the selected move.
                        continue
                    self.first_x_memory.update_memory(solution=Solution(goalfunctionvalue=goalval,
                                                                        move=swap,
                                                                        timestamp=ts))
                    improvement_found = True  # used as flag
                    # print statement will be deleted in future update
                    if verbose and False:
                        print("first x improving solutions:")
                        for sol in self.first_x_memory.memory:
                            print(sol)

                # when max amount x, of improvements is found, break the local neighbourhood search
                if self.first_x_memory.current_length >= self.first_x_memory.max_length:

                    final_solution = self.first_x_memory.best_solution

                    # tabulist and frequency memory will be updated by the selected move
                    self.tabu_list.update_tabulist(final_solution.move)
                    self.tabu_list.update_frequency_memory(final_solution.move)

                    # update the job sequence object
                    self.instance.update_job_df(self.instance.swap_jobs(swap=final_solution.move))
                    self.instance.update_goal()

                    # update best solution found
                    self.best_solution_memory.update_memory(solution=final_solution)
                    if verbose:
                        print('{} solutions found'.format(self.first_x_memory.current_length))
                        print('added solution {} to best solution memory'.format(final_solution))
                    # improvement found, no local optimum
                    local_optimum = False
                    max_improvements_reached = True
                    self.first_x_memory.clear_memory()  # ready for next iteration
                    break

            if improvement_found and not max_improvements_reached:
                # If less than x improving moves are found in local neighbourhood
                # Choose best solution from improving moves
                final_solution = self.first_x_memory.best_solution

                # tabulist and frequency memory will be updated by the selected move
                self.tabu_list.update_tabulist(final_solution.move)
                self.tabu_list.update_frequency_memory(final_solution.move)

                # update the job sequence object
                self.instance.update_job_df(self.instance.swap_jobs(final_solution.move))
                self.instance.update_goal()

                # update best solution memory
                self.best_solution_memory.update_memory(solution=final_solution)

                if verbose:
                    print('{} solutions found'.format(self.first_x_memory.current_length))
                    print('added solution {} to best solution memory'.format(final_solution))

                # improvement found, no local optimum
                local_optimum = False
                self.first_x_memory.clear_memory()  # ready for next iteration

            # case of local otpimum
            if local_optimum:
                if verbose:
                    print(50*'--')
                    print('LOCAL OPTIMUM')
                    print('Last solution found: {}'.format(self.best_solution_memory.last_solution()))
                    print('Best Global Solution encountered: {}'.format(self.best_solution_memory.best_solution))
                    print('Best Solution found in local neighbourhood: {}'.format(iteration_memory.best_solution))
                    print(50*'--')

                swap = iteration_memory.best_solution.move

                # update the tabulist and frequency memory with the selected move
                self.tabu_list.update_tabulist(swap)
                self.tabu_list.update_frequency_memory(swap)

                # update the job sequence object
                self.instance.update_job_df(self.instance.swap_jobs(swap=swap))
                self.instance.update_goal()

                self.best_solution_memory.update_memory(solution=iteration_memory.best_solution)
                if verbose:
                    print('Added Non-improving solution {} '
                          'to best solution memory'.format(iteration_memory.best_solution))

                self.first_x_memory.clear_memory()  # ready for next iteration, this should be empty?

        if verbose:
            print("Total runtime: ", time.time() - self.starttime)

        plotlabel = 'LS_I{}_X{}'.format(self.max_iter, self.x_improvements)
        if self.use_tabu_memory:
            plotlabel = 'TS_I{}_X{}_TL{}'.format(self.max_iter, self.x_improvements, self.tabu_length)
        self.best_solution_memory.plot_memory(self.starttime, label=plotlabel)

        # self.solution_memory.plot_memory(self.starttime, label='solution memory', custom_name='test')
        return self.instance


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
        self.showiterations = BooleanVar()
        btnrow = Frame(self)

        verbosebtn = Checkbutton(btnrow, text='Verbose?', variable=self.verbose,
                                 onvalue=True, offvalue=False)
        iterbtn = Checkbutton(btnrow, text='Show Iter?', variable=self.showiterations,
                              onvalue=True, offvalue=False)

        verbosebtn.select()
        iterbtn.select()

        btnrow.pack(fill=X)
        verbosebtn.pack()
        iterbtn.pack()

        removerow = Frame(self)
        removelabel = Label(removerow, text='Tabulist Remove Mechanism:')
        removerow.pack()
        removelabel.pack()
        self.remove_var = StringVar()
        remove_options = ['fifo', 'random']
        for opt in remove_options:
            Radiobutton(removerow, text=opt, command=self.onPress, variable=self.remove_var, value=opt).pack(anchor=W)
        self.remove_var.set('fifo')

        runbtn = Button(removerow, text='Run!', command=parent.quit).pack()

    def fetch_entries(self):

        for (fieldname, entry) in zip(self.fieldnames.keys(), self.entries):
            self.fieldnames.update({fieldname: int(entry.get())})
        return self.fieldnames

    def onPress(self):

        pick = self.remove_var.get()
        return pick
