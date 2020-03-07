from __future__ import print_function
import numpy as np
from subprocess import check_output
import sys


# create the output files:
run_min = 4900
run_max = 17700
step = 100
edges = np.arange(run_min,run_max,step)
file_array = []
for e in edges[:-1]:
    fname= './run_split_off/runnr_{}_{}.txt'.format(e,e+step)
    # get the pot and counts:
    output = check_output("/uboone/app/users/zarko/getDataInfo.py -v2 --file-list {}".format(fname), shell=True)
    #print(output)
    lines= output.split('\n')
    pot = lines[2].split()[7]
    ed1cnt =lines[2].split()[5]
    ext = lines[2].split()[0]
    print(e, pot, ed1cnt, ext)
