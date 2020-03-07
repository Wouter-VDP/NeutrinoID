import os
import subprocess
import sys

# if given a definition, it makes a filelist:

defname = sys.argv[1]
outputfile = sys.argv[2]
max_nfiles = int(sys.argv[3])

command = 'samweb list-files "defname: '+defname+'"'
pipe = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout
n = 0

with open(outputfile, 'w') as output_file:
    flist = pipe.read().split()
    while ((n < max_nfiles) and (n < len(flist)) ):
        filename = flist[n]
        command = 'samweb locate-file '+filename
        pipe2 = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE).stdout
        location = pipe2.read()
        if location == '':
            print('file deleted')
            continue
        location = location.split(":")[1].split("(")[0]
        output_file.write(location.rstrip()+"/"+filename+'\n')
        n += 1
