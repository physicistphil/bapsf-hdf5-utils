from Tkinter import *
import ttk
import h5py

f = h5py.File('/data/BAPSF_Data/Nonlinear_Alfven_Interactions/April17/7/07_45x45plane_B4B5B8_B850G_2antennaoverlap_A47_150V.hdf5')



root = Tk()
tree = ttk.Treeview(root)

def new_tree(d, item):
    if len(item.keys()) != 0:
        for iter in item:
            tree.insert(d, 1, "", text=iter)

for iter in f:
    name_dir = f[iter]
    directory = tree.insert("", 1, "", text=iter)
    new_tree(directory, name_dir)

tree.pack()
root.mainloop()
