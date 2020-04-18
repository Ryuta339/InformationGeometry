import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anime
from channel import Channel


p = np.zeros ((2,))
p[0] = 0.6
p[1] = 0.4
w = np.array ([[2/3,1/2],[1/3,1/2]])
ch = Channel (w)

ims = []
fig = plt.figure ()

im1 = plt.plot ([0,1],[1,0],label="Manifold")
im2 = plt.plot (p[0],p[1],"o",color="red")
ims.append (im1+im2)

for _ in range (2000):
    p = ch.nextProb (p)
    im2 = plt.plot (p[0], p[1], "o", color="red")
    ims.append (im1+im2)

ani = anime.ArtistAnimation (fig, ims, interval=50)
ani.save ("anim2.mp4",  writer="ffmpeg")

plt.show ()

