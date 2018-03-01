from tkinter import *


def fetch(entries, fields):
    for entry, fieldname in zip(entries, fields.keys()):
        print(fieldname, entry.get())
        fields.update({fieldname: entry.get()})
    return fields


def makeform(root, fields):
    entries = []
    for fieldname, fieldvalue in zip(fields.keys(), fields.values()):
        row = Frame(root)
        lab = Label(row, width=10, text=fieldname)
        ent = Entry(row)
        ent.insert(0, fieldvalue)
        row.pack(side=TOP, fill=X)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT)
        entries.append(ent)
    return entries


fieldict = {'max_iter': 20}
root = Tk()
ents = makeform(root, fieldict)
btn = Button(root, text='Run!', command=root.quit).pack()
root.mainloop()

max_iter = fetch(ents, fieldict)
