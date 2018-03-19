import numpy as np
import pandas as pd
import sys
from tkinter import *
"""
This script will generate an instance for the single machine scheduling case.
"""

# regular entries
entry_options = {
    'replicates': 5,
    'n': 80,
    'm': 1,
    'processing lower': 1,
    'processing upper': 500,
    'setuptime lower': 0.25,
    'setuptime upper': 0.75,
}

# checkbuttons
d = 1  # handles independence issues ??

# radiobuttons
# 1: group fractions

G = {
    'L': .1,  # low option
    'M': .3,  # medium option
    'H': .9   # high option
}

# default value
nG = entry_options['n'] * G['L']  # amount of job families

# 2: Tardiness factor

T = {
    'L': .3,
    'H': .6
}

# 3: due date spread parameter

W = {
    'L': .5,
    'H': 2.5
}


class Generator(Frame):

    def __init__(self, parent, general_entries, **options):

        self.parent = parent
        Frame.__init__(self, parent, **options)
        self.general_entries = general_entries  # dictionary with labels and values
        self.pack()
        self.entries = []

        for (fieldname, fieldvalue) in self.general_entries.items():
            row = Frame(self)
            lab = Label(row, width=15, text=fieldname)
            ent = Entry(row)
            ent.insert(0, fieldvalue)
            row.pack(side=TOP, fill=X)
            lab.pack(side=LEFT)
            ent.pack(side=RIGHT)
            ent.focus()
            ent.bind('<Return>', self.submit)
            self.entries.append(ent)

        # radiobutton row for tardiness factor T
        row = Frame(self)
        label = Label(row, width=15, text='tardiness (T)')
        container = Frame(row)
        self.tardiness_factor = DoubleVar()
        options = [0.0, 0.5, 2.5]

        row.pack(expand=YES, fill=BOTH)
        label.pack(side=LEFT)
        container.pack(side=RIGHT, fill=X, expand=YES)
        for opt in options:
            Radiobutton(container, text=opt, command=(lambda: self.onPress(self.tardiness_factor)),
                        variable=self.tardiness_factor, value=opt).pack(side=LEFT, expand=YES)
        self.tardiness_factor.set(0.5)

        # radiobutton row for spread parameter R
        row = Frame(self)
        label = Label(row, width=15, text='spread (R)')
        container = Frame(row)
        self.spread_param = DoubleVar()
        options = [0.0, 0.3, 0.6]

        row.pack(expand=YES, fill=BOTH)
        label.pack(side=LEFT)
        container.pack(side=RIGHT, expand=YES, fill=X)
        for opt in options:
            Radiobutton(container, text=opt, command=(lambda: self.onPress(self.spread_param)),
                        variable=self.spread_param, value=opt).pack(side=LEFT, expand=YES)
        self.spread_param.set(0.3)

        # radiobutton row for grouping parameter G
        row = Frame(self)
        label = Label(row, width=15, text='grouping (g)')
        container = Frame(row)
        self.group_param = DoubleVar()
        options = [0.1, 0.3, 0.9]

        row.pack(expand=YES, fill=BOTH)
        label.pack(side=LEFT)
        container.pack(side=RIGHT, fill=X, expand=YES)
        for opt in options:
            Radiobutton(container, text=opt, command=(lambda: self.onPress(self.group_param)),
                        variable=self.group_param, value=opt).pack(side=LEFT, expand=YES)
        self.group_param.set(0.1)

        # final row to submit
        submitrow = Frame(self)
        quitbtn = Button(submitrow, text='Quit', command=self.parent.quit)
        submitbtn = Button(submitrow, text='Submit', command=self.submit)

        submitrow.pack()
        quitbtn.pack(side=LEFT, expand=YES)
        submitbtn.pack(side=RIGHT, expand=YES)

    def submit(self, event=None):
        print('I should submit something')
        for entry, name in zip(self.entries, self.general_entries.keys()):
            print(name, entry.get())
            self.general_entries.update({name: entry.get()})

        self.parent.quit()

    def onPress(self, var):
        print('pressed on something: ', var.get())


root = Tk()
root.title('Local Search Initializer - Single Machine Job Shop')
GUI = Generator(parent=root, general_entries=entry_options)  # start the GUI

# Put GUI Window on top of current window
root.lift()
root.after_idle(root.attributes, '-topmost', False)
root.mainloop()

print('the final entries were:\n')
for entry in GUI.entries:
    print(entry.get())

print(GUI.general_entries)