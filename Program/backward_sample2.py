import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt
from matplotlib import animation as anime
from channel import Channel
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d


def nextProb (channel, p):
    p0 = p[0]
    p1 = p[1]
    prob = np.array ([p0,p1,1.-p0-p1])
    return channel.nextProb (prob)[0:2]

w = np.array ([[1/2,1/4,1/4],[1/4,1/2,1/4],[1/4,1/4,1/2]])
ch = Channel (w)
p = np.array ([0.0001,0.0001]);

ims = []
fig = plt.figure ()
ax = fig.add_subplot (111, projection='3d')

x = [1, 0, 0]
y = [0, 1, 0]
z = [0, 0, 1]
poly = list (zip (x,y,z))
ax.add_collection3d(art3d.Poly3DCollection([poly],color='c',alpha=0.4))
ax.set_xlabel("p[0]")
ax.set_ylabel("p[1]")
ax.set_zlabel("p[2]")
ax.view_init(elev=30, azim=30)

im = ax.plot ([p[0]], [p[1]], [1.-p.sum()], "o", color="red")
ims.append (im)

for _ in range (100):
    sol = opt.root (lambda prev: nextProb (ch, prev) - p, p);
    p = sol.x[0:2]
    im = ax.plot ([p[0]], [p[1]], [1.-p.sum()], "o", color="red")
    ims.append (im)

ani = anime.ArtistAnimation (fig, ims, interval=50)
ani.save ("anim_3d_2.mp4",  writer="ffmpeg")

plt.show ()
