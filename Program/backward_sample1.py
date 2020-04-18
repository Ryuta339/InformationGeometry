import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt
from matplotlib import animation as anime
from channel import Channel


def nextProb0 (channel, p):
    p0 = p[0]
    prob = np.array ([p0, 1-p0])
    return channel.nextProb (prob)[0]

p = 0.0001
w = np.array ([[2/3,1/2],[1/3,1/2]])
ch = Channel (w)

ims = []
fig = plt.figure ()

im1 = plt.plot ([0,1],[1,0],label="Manifold")
im2 = plt.plot (p,1.-p,"o",color="red")
ims.append (im1+im2)

for _ in range (500):
    sol = opt.root (lambda prev: nextProb0 (ch, prev) - p, p)
    p = sol.x[0]
    im2 = plt.plot (p,1.-p,"o",color="red")
    ims.append (im1+im2)

ani = anime.ArtistAnimation (fig, ims, interval=50)
ani.save ("anim3.mp4", writer="ffmpeg")

plt.show ()
