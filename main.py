from classes import *

import re
# custom sorter for os.listdir data


def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


def local_search_initializer(use_gui=True):

    # use for the entry values

    metaparameters_defaults = {'max_iter': 15,
                      'x_improvements': 5,
                      'tabulength': 30}

    # use for the checkbuttons
    verbosity = False  # shows best solution per iteration
    show_iter = True  # shows the current iteration
    show_img = True  # makes a plot of the best solution for each iteration
    tabu_remove = 'fifo'  # the move removal procedure for the tabulist

    if use_gui:
        root = Tk()
        root.title('Local Search Initializer - Single Machine Job Shop')
        GUI = LocalSearchGui(parent=root, fieldnames=metaparameters_defaults)  # start the GUI

        # Put GUI Window on top of current window
        root.lift()
        root.after_idle(root.attributes, '-topmost', False)
        root.mainloop()

        # fetch the inputs for the metaprameters
        metaparameters = GUI.fetch_entries()
        verbosity = GUI.verbose.get()
        show_iter = GUI.showiterations.get()
        show_img = GUI.show_img.get()
        tabu_remove = GUI.remove_var.get()

        metaparameters.update({'verbose': verbosity,
                               'show_iter': show_iter,
                               'show_img': show_img,
                               'tabu_remove': tabu_remove})

    else:
        metaparameters = metaparameters_defaults.update({'verbose': verbosity,
                                                         'show_iter': show_iter,
                                                         'show_img': show_img,
                                                         'tabu_remove': tabu_remove})

    return metaparameters


# will be depreciated in future
def local_search_algorithm(metaparamdict, use_gui=True):

    if use_gui:
        root = Tk()
        GUI = LocalSearchGui(parent=root, fieldnames=metaparamdict)
        root.lift()
        root.after_idle(root.attributes, '-topmost', False)
        root.mainloop()
        metaparameters = GUI.fetch_entries()
        verbosity = GUI.verbose.get()
        showiter = GUI.showiterations.get()
        tabu_remove = GUI.remove_var.get()

    else:
        metaparameters = metaparamdict
        verbosity = True
        showiter = True
        tabu_remove = 'fifo'

    # read instance and create jobschedule object
    instance = JobSchedule(pathname='instance.csv')

    # create local seach algorithm object
    algorithm = LocalSearch(instance, max_iter=metaparameters['max_iter'],
                            x_improvements=metaparameters['x_improvements'],
                            tabulength=metaparameters["tabulength"],
                            tabu_remove=tabu_remove)

    # solve the jobschedule
    instance = algorithm.solve(verbose=verbosity, show_iter=showiter)

    # output solution path
    solution = algorithm.best_solution_memory.create_solution_df(algorithm.starttime)

    return solution, instance


def local_search_intantiator(initializer_dict, instance_):

    _algorithm = LocalSearch(jobschedule=instance_,
                             max_iter=initializer_dict['max_iter'],
                             x_improvements=initializer_dict['x_improvements'],
                             tabulength=initializer_dict['tabulength'],
                             tabu_remove=initializer_dict['tabu_remove'])

    return _algorithm


if __name__ == '__main__':

    # solve a problem flow:
    # 1. get the algorithm's metaprameters
    initializer = local_search_initializer()
    for instancepath in sorted_aphanumeric(os.listdir('instances'))[:2]:

        # 2. get the Jobschedule object for the problem
        instance = JobSchedule(pathname='instances/'+instancepath)

        # 3. instantiate the algorithm
        algorithm = local_search_intantiator(initializer, instance)

        # 4. solve the instance
        solved_instance = algorithm.solve(verbose=initializer['verbose'],
                                          show_iter=initializer['show_iter'],
                                          show_img=initializer['show_img'])

        # 5. create a dataframe with found solutions and timestamps
        solution = algorithm.best_solution_memory.create_solution_df(algorithm.starttime)

        # 6. write this solutions to a destination folder
        try:
            os.mkdir('solutionfolder/{}'.format(instancepath.split('.')[0]))
        except:
            # then the folder already exists
            pass

        solution.to_csv('solutionfolder/{}/{}'.format(instancepath.split('.')[0], algorithm.solution_path))

    quit()

    metaparam1 = {"max_iter": 20, "x_improvements": 5, 'tabulength': 15}
    sol1, instance = local_search_algorithm(metaparam1, use_gui=True)
    sol1.to_csv('solutions/sol3.csv')

    # metaparam2 = {"max_iter": 40, "x_improvements": 5, 'tabulength': None}
    # sol2 = local_search_algorithm(metaparam2, use_gui=False)
    # sol2.to_csv('solutions/sol4.csv')