import numpy as np
import json

'''
file = './ParameterOne.txt'
fp = open(file)
data = fp.read()
fp.close()
adata = {"loss": 2.3303, "step": 0, "iter": 0, "accuracy": 0.0625}
klist = adata.keys()
print "klist = ",klist
outdata = []
while(len(data) > 0):
    tlist = []
    start = data.index('{')
    end   = data.index('}')
    strdata = data[start:end+1]
    # print "strdata = ",strdata
    adict = json.loads(strdata)
    for k in klist:
        tlist.append(adict[k])
    outdata.append(tlist)
    data = data[end+1:]
    print "tlist = ",tlist
print "-"*80

outdata = np.array(outdata)
print "outdata = ",outdata.shape
np.save('./npyfile/train_out',outdata)
'''



outdata = np.load('./npyfile/train_out.npy')
print "outdata = ",outdata.shape
# print outdata[100:110]
print "-"*80
shape = outdata.shape
# data cols  ['loss', 'step', 'iter', 'accuracy']
iterTimes0 = []
step = 0
epoch = 99
for i in xrange(shape[0]):
    if int(outdata[i,2]) == epoch:
        iterTimes0.append(outdata[i])
    else:
        pass
iterTimes0 = np.array(iterTimes0)
print "iterTimes0.shape = ",iterTimes0.shape

from matplotlib import pyplot as plt
plt.plot(iterTimes0[:,0],'ro')
plt.plot(iterTimes0[:,3],'bo')
plt.plot(iterTimes0[:,0],'r-')
plt.plot(iterTimes0[:,3],'b-')
plt.legend(['loss','accuracy'])
plt.grid(True)
plt.ylabel('value')
plt.xlabel('step in epoch x10')
plt.title('Accuracy And Loss In epoch')
plt.savefig('./images/epoch-99')
plt.show()
'''
length = 39  #epoch = 0,1,2,3
shape1 = ['ro','bo','go','mo']
shape2 = ['r-','b-','g-','m-']

j = 0 # j = 0 loss; j = 3 accuracy
legend = []
epochlist = [0,1,2,3]
for i in epochlist:
    start = i*length
    end = start + length
    data = outdata[start:end,:]
    plt.plot(data[:,j],shape1[i])
    # plt.plot(iterTimes0[:,3],'r-')
    legend.append('epoch = '+str(i))
plt.legend(legend)
for i in epochlist:
    start = i*length
    end = start + length
    data = outdata[start:end,:]
    # plt.plot(iterTimes0[:,3],'ro')
    plt.plot(data[:,j],shape2[i])
plt.grid(True)
plt.xlabel('step in epoch x10')

plt.ylabel('accuracy')
plt.title('Accuracy In epoch')
plt.savefig('./images/accuracy-epoch')
plt.show()

plt.ylabel('loss')
plt.title('Loss In epoch')
plt.savefig('./images/loss-epoch')
plt.show()
'''