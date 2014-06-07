#!/usr/bin/python
import re
import subprocess
import os
import shutil
import sys
import locale
import math
import copy

# CONVERGENCE IN TIME #
loops = 8
init_time_mult=2
time_power=2;
space_mult=25;
final_time=1;
optimization_times_time_error=1;

locale.setlocale(locale.LC_NUMERIC, "")
def format_num(num):
    """Format a number according to given places.
    Adds commas, etc. Will truncate floats into ints!"""

    try:
        inum = int(num)
        return locale.format("%.*f", (2, num), True)

    except (ValueError, TypeError):
        return str(num)

def get_max_width(table, index):
    """Get the maximum width of the given column index"""
    return max([len(format_num(row[index])) for row in table])

def pprint_table(out, table):
    """Prints out a table of data, padded for alignment
    @param out: Output stream (file-like object)
    @param table: The table to print. A list of lists.
    Each row must have the same number of columns. """
    col_paddings = []

    for i in range(len(table[0])):
        col_paddings.append(get_max_width(table, i))

    for row in table:
        # left col
        print >> out, row[0].ljust(col_paddings[0] + 1),
        # rest of the cols
        for i in range(1, len(row)):
            col = format_num(row[i]).rjust(col_paddings[i] + 2)
            print >> out, col,
        print >> out

out = sys.stdout

src = 'default.prm'
dst = 'runtime.prm'
shutil.copyfile(src,dst);

subsystems = 'fluid', 'structure'
variables=(('vel', 'press'),('displ', 'vel'))
show_errors = (2,1,2,1) # L2 and H1 for fluid velocity, only L2 for fluid pressure, etc...
error_names = ('dofs','L2','H1')
errors=[[]]
for i in range(len(subsystems)):
    for j in range(len(variables)):
        for k in range(show_errors[i*len(subsystems)+j]):
            errors[0].append(subsystems[i]+'.'+variables[i][j]+'.'+error_names[0])
            errors[0].append(subsystems[i]+'.'+variables[i][j]+'.'+error_names[k+1])

for i in range(loops):
    # RESET TIME
    n_steps = init_time_mult * pow(time_power,i);

    # READ IN DEFAULT.PRM AND REPLACE TIME STEPS LINE 
    f_in = open('default.prm')
    lines = f_in.readlines()
    f_in.close()
    for key, line in enumerate(lines):
        if line[0]!='#':
            success = re.search("number of time steps", line)
            if success:
                lines[key] = 'set number of time steps  = ' + str(n_steps)
            success = re.search("jump tolerance", line)
            if success:
                lines[key] = 'set jump tolerance  = ' + str(float(optimization_times_time_error*final_time)/n_steps)
            success = re.search("nx", line) or re.search("ny", line)
            if success:
                objMatch=re.match( r'(.* = )(.*)', line, re.M|re.I)
                lines[key] = objMatch.group(1) + str(space_mult)
            objMatch=re.match( r'(.* convergence method .* = )(.*)', line, re.M|re.I)
            if objMatch:
                lines[key] = objMatch.group(1) + 'time'


    # WRITE OVER RUNTIME.PRM
    with open(dst, 'w') as file:
        file.writelines( lines )

    # RUN THE PROGRAM USING THE RUNTIME.PRM    
    subprocess.call(['./FSI_Project', dst])

    # COLLECT ERROR INFORMATION 
    f_in = open('errors.dat')
    lines = f_in.readlines()
    f_in.close()
    errors.append([0 for j in range(len(errors[0]))])
    for line in lines:
        for key, name in enumerate(errors[0]):
            objMatch=re.match( r'^'+re.escape(name)+':(.*)', line, re.M|re.I)
            if objMatch:
                errors[i+1][key] = objMatch.group(1)     

count_down = 0
convergence_rate = copy.deepcopy(errors)
convergence_rate[0][0]="1/dt"
for i in range(len(subsystems)):
    for j in range(len(variables)):
        for k in range(show_errors[i*len(subsystems)+j]):
            convergence_rate[1][count_down+1]='-'
            for l in range(loops-1):
                convergence_rate[l+2][count_down+1]=math.log(float(errors[l+2][count_down+1])/float(errors[l+1][count_down+1]),10)/math.log(float(errors[l+1][count_down])/float(errors[l+2][count_down]),10)
            count_down+=2
convergence_rate = map(list, zip(*convergence_rate))
errors = map(list, zip(*errors))

#print errors
pretty_convergence_rate=[]
for i in range(len(convergence_rate)):
    if i<2 or i%2==1:
        pretty_convergence_rate.append(convergence_rate[i])
pretty_convergence_rate = map(list, zip(*pretty_convergence_rate))
pprint_table(out, pretty_convergence_rate)
