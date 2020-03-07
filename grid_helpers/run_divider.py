from __future__ import print_function
import numpy as np
import subprocess
import sys


file_name = "run2/run2_on_D2.txt"

# read some input files:
file_list = [line.rstrip().split('/')[-1] for line in open(file_name, 'r')]
print('Number of files:',len(file_list))

# create the output files:
run_min = 4900
run_max = 17700
step = 100
edges = np.arange(run_min,run_max,step)
file_array = []
for e in edges[:-1]:
    if 'on' in file_name:
        file_array.append(open('run_split_on/runnr_{}_{}.txt'.format(e,e+step),'a+'))
    elif 'off' in file_name:
        file_array.append(open('run_split_off/runnr_{}_{}.txt'.format(e,e+step),'a+'))
    else:
        print('Sample is not data?')
        
#loop over files and query runnr:
for i,root_file in enumerate(file_list):
    command = subprocess.Popen(["samweb",  "get-metadata", root_file], stdout=subprocess.PIPE)
    output  = subprocess.check_output(('grep', 'Runs'), stdin=command.stdout)
    this_run= output.decode("utf-8").strip().split()[1].split('.')[0]
    f_index = np.digitize(this_run,edges)-1
    print('{}/{}\t(run: {}, index: {})'.format(i, len(file_list), this_run, f_index))
    file_array[f_index].write(root_file+'\n')
    
# close the output files:
a = [f.close() for f in file_array]

