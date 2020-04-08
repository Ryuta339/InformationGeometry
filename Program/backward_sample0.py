import numpy as np
from scipy import optimize as opt
from channel import Channel


w = np.array ([[2/3,1/3],[1/3,2/3]]);
ch = Channel (w)
p = np.ones ((2,)) / 2

x0 = np.zeros ((2,))
x0[0] = 1.

sol = opt.root (lambda prob: ch.nextProb (prob) - p, x0)
prev = sol.x

print (prev / prev.sum ())



w2 = np.array ([[2/3,1/2],[1/3,1/2]]);
ch2 = Channel (w2)
p2 = np.zeros ((2,))
p2[0] = np.power (5*49/2048, 1/3)
p2[1] = np.sqrt (35) / 12
p2 = p2 / p2.sum ()

sol = opt.root (lambda prob: ch2.nextProb (prob) - p2, x0);
prev2 = sol.x
print (prev2 / prev2.sum ())
