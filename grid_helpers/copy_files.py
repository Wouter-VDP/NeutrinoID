import os
from subprocess import STDOUT, check_output

names = ['on','off']
run=2
         
for projectname in names:
    print("\nStart with project:",projectname)
    version = "v08_02_00"
    analyser = "NueCC"
    outdir = "/uboone/data/users/wvdp/reco2_hist_files_feb202/run{}/{}/".format(run,projectname)

    os.system("kx509")
    os.system("voms-proxy-init -noregen -voms fermilab:/fermilab/uboone/Role=Analysis")
    xrdroot = "xroot://fndca1.fnal.gov/pnfs/fnal.gov/usr/"

    filelist = [line.rstrip() for line in open('/uboone/app/users/wvdp/Tools/NuFilter_Feb2020/run{}/run_{}_{}_hist.txt'.format(run,run,projectname))]
    print('Files collected: ',len(filelist))
    counter = 0
    existed = 0
    timed_out = 0

    if not os.path.isdir(outdir):
	    print("Output directory did not exist, creating it.")
	    os.system("mkdir "+outdir)

    for i,fname in enumerate(filelist):
	    source = xrdroot+"/".join(fname.split("/")[2:])
	    destination = outdir+fname.split("/")[-1]
	    if os.path.isfile(destination): 
		    print("File exists already, skipped.")
		    existed+=1
		    continue
	    command = "xrdcp "+source+" "+destination
	    try:
	        output = check_output(["xrdcp", source, destination], stderr=STDOUT, timeout=1)
	        #print(output)
	        print('{}({})/{}, fname {}'.format(i,counter+existed,len(filelist),fname.split("/")[-1]))
	        counter+=1
	    except:
	        print("Took too long, skipping")
	        timed_out+=1
	        
print(len(filelist),'files in the list')
print(counter,"files were copied")
print(existed,"files existed already")
print(timed_out,"files timed-out")
