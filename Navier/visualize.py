#!/usr/bin/python
import sys
import csv
import numpy as np
import pylab

''' This function is meant to plot x and y displacement as well as lift and drag for a given input file. 
usage:
    >> visualize.py filename

where the file is formatted:
time x_displacement y_displacement lift drag
'''

assert len(sys.argv) is 2, "Improper usage. Give a filename as argument."

def get_values(filename):
    f = open(filename)
    csv_file = csv.reader(f)
    all_entries = []
    for row in csv_file:
        all_entries.append(row[0].split(' '))
    return zip(*all_entries)

def make_plot(independent, dependent, plot_title):
    pylab.plot(independent, dependent)
    pylab.xlabel('time (s)')
    pylab.ylabel(plot_title)
    pylab.title('Tracking over Time')
    pylab.grid(True)
    #savefig("test.png")
    pylab.show()


all_entries = get_values(str(sys.argv[1]))
plot_titles = ['Time','X-Displacement','Y-Displacement','Lift','Drag']
t = all_entries[0]
for index in range(1,5):
    make_plot(t, all_entries[index], plot_titles[index])


