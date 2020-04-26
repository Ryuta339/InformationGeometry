import numpy as np
from scipy import optimize as opt
from channel import Channel

def nextProb0 (channel, p):
    p0 = p[0]
    prob = np.array ([p0, 1-p0])
    return channel.nextProb (prob)[0]

w = np.array ([[2/3,1/3],[1/3,2/3]]);
ch = Channel (w)
p = np.ones ((2,)) / 2

x0 = np.zeros ((3,))
x0[0] = 1.

sol = opt.root (lambda prob: calcPrev (ch, prob, p), x0)
prev = sol.x
print (prev)



w2 = np.array ([[2/3,1/2],[1/3,1/2]]);
ch2 = Channel (w2)
p2 = np.zeros ((2,))
p2[0] = np.power (5*49/2048, 1/3)
p2[1] = np.sqrt (35) / 12
p2 = p2 / p2.sum ()

sol = opt.root (lambda prob: calcPrev (ch2, prob, p2), x0);
prev2 = sol.x
print (prev2)
