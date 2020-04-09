import numpy as np

'''
    W is a 2x2 matrix
    W[col,row] = W(col|row)
'''
class Channel:
    def __init__ (self, w):
        self.w = w

    def goThrough (self, prob):
        return self.w @ prob

    def kullbackLeibler (self, r):
        if r.shape != (2,1):
            r = r.reshape ((2,1))
        tmp = self.w * np.log (self.w / r)
        return tmp.sum (axis=0)


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

r = ch.goThrough (p);
p2 = p * np.exp (- ch.kullbackLeibler (r))
p2 = p2 / p2.sum()

print (p2)
