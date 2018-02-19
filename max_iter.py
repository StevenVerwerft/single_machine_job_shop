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
        ent.focus()
        ent.bind('<Return>', (lambda event: root.quit()))
        entries.append(ent)
    return entries


fieldict = {'max_iter': 5}
root = Tk()
ents = makeform(root, fieldict)
img_show = IntVar()
imgcheckbtn = Checkbutton(root, text='IMG', variable=img_show, onvalue=1, offvalue=0)
imgcheckbtn.select()
imgcheckbtn.pack()

verbose = IntVar()
verbosebtn = Checkbutton(root, text='Verbose?', variable=verbose, onvalue=1, offvalue=0)
verbosebtn.select()
verbosebtn.pack()

btn = Button(root, text='Run!', command=root.quit).pack()
root.mainloop()
max_iter = fetch(ents, fieldict)
