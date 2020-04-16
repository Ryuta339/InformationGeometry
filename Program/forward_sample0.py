import numpy as np
from channel import Channel

#p = np.ones ((2,)) / 2
#w = np.array ([[2,1],[1,2]]) / 3
#ch = Channel (w);
#
#r = ch.goThrough (p);
#p2 = p * np.exp (- ch.kullbackLeibler (r))
#
#p2 = p2 / p2.sum ()
#
#print (p2)
#



p = np.ones ((2,)) / 2
w = np.array ([[2/3,1/2],[1/3,1/2]])
ch = Channel (w)

# r = ch.goThrough (p);
# p2 = p * np.exp (- ch.kullbackLeibler (r))
# p2 = p2 / p2.sum()
p2 = ch.nextProb (p)

print (p2)
