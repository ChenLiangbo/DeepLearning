#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import os


outdir = './npyfile/'
def getNumpyFromNohupOut(filename):
    fp = open(filename,'rb')
    lines = fp.readlines()
    fp.close()

    data = []
    count = 0
    for line in lines:
        dline = []
        # print "line = ",line
        if len(line) > 40:
            line = line.split(',')
            # print "line = ",line
            for e in line:
                # print "e = ",e
                e = e.split('=')
                # print "e = ",float(e[1])
                # print "-"*80
                try:
                    dline.append(float(e[1]))
                
                except:
                	print "[exception] count = ",count
            # print "dline = ",dline
            if len(line) != 4:
                print "line is not 4 line = ",line
                break
            data.append(dline)
        count =  count + 1
        # break
    print "count = ",count
    
    data = np.array(data)
    return data

datalist = []
legend = []
infer = {"One":1,'Tow':2,'Three':3,'Four':4,'Five':5,'Six':6}
nohupdir = './file/'
filelist = os.listdir(nohupdir)
for f in filelist:
    if 'nohup' in f:
        filename = nohupdir + f
        print "filename = ",filename
        for k in infer:
            if k in f:
                legend.append(infer[k])
        data = getNumpyFromNohupOut(filename)
        print "data.shape = ",data.shape
        datalist.append(data)
print "-"*80
print "legend = ",legend
print "data.shape = ",(len(datalist),data.shape)

iterTimesList = []
s_epoch = 0
e_epoch = 1
for i in xrange(len(datalist)):
    iterTimes0 = []
    data = datalist[i]
    dshape = data.shape
    print "dshape = ",dshape
    for m in xrange(dshape[0]):
        if (data[m,0] >= s_epoch) and (data[m,0] < e_epoch):
            iterTimes0.append(data[m,:])
    iterTimes0 = np.array(iterTimes0)
    iterTimesList.append(iterTimes0)
print "iterTimesList = ",(len(iterTimesList),iterTimesList[0].shape)
shapes1 = ['r*','b*','g*','y*','m*','c*']
shapes2 = ['r-','b-','g-','y-','m-','c-']

for i in xrange(len(legend)):
    legend[i] = 'NN-' + str(legend[i])


imagedir = './images/'
from matplotlib import pyplot as plt

select = 2
for i in range(len(iterTimesList)):
    iterTimes0 = iterTimesList[i]
    x = range(iterTimes0.shape[0])
    print "x = ",(i,len(x),iterTimes0.shape)
    plt.plot(x,iterTimes0[:,select],shapes1[i])
plt.legend(legend)

for i in range(len(iterTimesList)):
    iterTimes0 = iterTimesList[i]
    x = range(iterTimes0.shape[0])
    plt.plot(x,iterTimes0[:,select],shapes2[i])

plt.grid(True)
plt.ylabel('loss value')
plt.xlabel('step in epoch x10')
plt.title(' Loss In epoch Each NN ' + str(s_epoch) + ' - ' + str(e_epoch))
plt.savefig(imagedir + 'Loss-epoch-' + str(e_epoch))
plt.show()

# for i in xrange(5):

# np.save('plotNohupout',data)

'''

# i = 0,j = 10, loss = 3.158267, accuracy = 0.171875 

data = np.load('plotNohupout.npy')
print "data = ",data.shape
imagedir = './images/'
dshape = data.shape

epoch = 2
iterTimes0 = []
for i in xrange(dshape[0]):
    if data[i,0] < epoch:
        iterTimes0.append(data[i,:])
iterTimes0 = np.array(iterTimes0)
print "iterTimes0.shape = ",iterTimes0.shape


from matplotlib import pyplot as plt
plt.plot(iterTimes0[:,2],'ro')
plt.plot(iterTimes0[:,3],'bo')
plt.plot(iterTimes0[:,2],'r-')
plt.plot(iterTimes0[:,3],'b-')
plt.legend(['loss','accuracy'])
plt.grid(True)
plt.ylabel('value')
plt.xlabel('step in epoch x10')
plt.title('Accuracy And Loss In epoch < ' + str(epoch))
plt.savefig(imagedir + 'epoch-' + str(epoch))
plt.show()
'''