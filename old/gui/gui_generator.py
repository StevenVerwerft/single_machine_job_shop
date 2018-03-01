from tkinter import *
import sys

fieldnames = {'n': 80,
              'p_l': 1,
              'p_u': 500}


def fetch(entries, fields):
    for entry, fieldname in zip(entries, fields.keys()):
        print(fieldname, entry.get())
        fields.update({fieldname: entry.get()})
    return fields


def makeform(root, fields):
    entries = []
    for fieldname, fieldvalue in zip(fields.keys(), fields.values()):
        row = Frame(root)
        lab = Label(row, width=5, text=fieldname)
        ent = Entry(row)
        ent.insert(0, fieldvalue)
        row.pack(side=TOP, fill=X)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries.append(ent)
    return entries


if __name__ == '__main__':
    root = Tk()
    ents = makeform(root, fieldnames)
    Button(root, text='submit', command=fetch(ents, fieldnames)).pack()
    root.mainloop()

# win = Frame()
# win.pack()
#
# Label(win, text='Submit instance settings \n').pack()
#
# ent_n = Entry(win)
# ent_n.insert(0, 80)
# ent_n.pack()
#
# ent_p_l = Entry(win)
# ent_p_l.insert(0, 1)
# ent_p_l.pack()
#
# ent_p_u = Entry(win)
# ent_p_u.insert(0, 500)
# ent_p_u.pack()
#
# btn_submit = Button(win, text='Submit', command=fetch_all).pack()
#
# #ent.focus()
# #ent.bind('<Return>', (lambda event: fetch()))
#
# Button(win, text='quit', command=win.quit).pack()
#
# win.mainloop()
#
# n, p_l, p_u = fetch_all()
# print(n, p_l, p_u)