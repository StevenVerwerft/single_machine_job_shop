from classes import *


def main():

    metaparamdict = {"max_iter": 5, "x_improvements": 15}

    root = Tk()
    GUI = LocalSearchGui(parent=root, fieldnames=metaparamdict)
    root.lift()
    root.after_idle(root.attributes, '-topmost', False)
    root.mainloop()
    metaparameters = GUI.fetch_entries()

    print(metaparameters)
    quit()

    instance = JobSchedule(pathname='instance.csv')
    algorithm = LocalSearch(instance, max_iter=metaparameters['max_iter'],
                            x_improvements=metaparameters['x_improvements'])

    algorithm.solve(verbose=GUI.verbose.get())

    solution = algorithm.best_solution_memory.create_solution_df(algorithm.starttime)
    print(solution.head(50))


if __name__ == '__main__':
    main()
