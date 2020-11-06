import numpy as np
from RPM import *
import matplotlib

matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import csv
import sys

if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

root = Tk.Tk()
root.wm_title("Red multicapa")

f = Figure(figsize=(9, 6), dpi=100)

a = f.add_subplot(111)

a.set_xlim([-2, 2])
a.set_ylim([-2, 2])
a.grid(True)

canvas = FigureCanvasTkAgg(f, master=root)
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

def readFile(name):
    stream = []
    with open(name, newline='') as csvarchivo:
        reader = csv.reader(csvarchivo)
        for row in reader:
            newrow = [float(item) for item in row]
            stream.append(newrow)
    return stream

X = np.array(readFile('inputs.csv'), dtype=float)
y = np.array(readFile('outputs.csv'), dtype=float)

epochs = 10000
epoch = 0
presition = 0.01
E = 1.0
lr = 0.3

for i in range(len(y)):
    c = 'slateblue'
    if(y[i][0] > 0):
        c = 'slategray'
    a.scatter(X[i][0],X[i][1], c = c, s = 75)

def learningRate(event):
    global lr
    
    try:
        lr = float(lrEntry.get())
    except ValueError:
        print('El numero ingresado no es correcto')

lrLabel = Tk.Label(root, text="Learning rate")
lrLabel.pack(side='left')
lrEntry = Tk.Entry(root)
lrEntry.insert(0, lr)
lrEntry.bind("<Return>", learningRate)
lrEntry.pack(side='left')

def maxEpochs(event):
    global epochs
    try:
        epochs = float(epochsEntry.get())
    except ValueError:
        print('El numero ingresado no es correcto')

epochsLabel = Tk.Label(root, text="Maximo de epocas")
epochsLabel.pack(side='left')
epochsEntry = Tk.Entry(root)
epochsEntry.insert(0, epochs)
epochsEntry.bind("<Return>", maxEpochs)
epochsEntry.pack(side='left')

def train():
    global epoch, E
    NN = NeuralNetwork(lr)

    while epoch < epochs and E > presition:

        E = np.mean(np.square(y - NN.feedForward(X)))
        NN.train(X, y)
        epoch = epoch +1

    print("Epoca: " + str(epoch))

    print("Predicted Output: " + str(NN.feedForward(X)))
 
trainButton = Tk.Button(root, text="Entrenar", command=train)
trainButton.pack(side='left')

Tk.mainloop()