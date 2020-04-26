import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anime
from channel import Channel
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d


p = np.zeros ((3,))
p[0] = 0.3
p[1] = 0.4
p[2] = 0.3
w = np.array ([[1/2,1/4,1/4],[1/4,1/2,1/4],[1/4,1/4,1/2]])
ch = Channel (w)

ims = []
fig = plt.figure ()
ax = fig.add_subplot (111, projection='3d')

x = [1, 0, 0]
y = [0, 1, 0]
z = [0, 0, 1]
poly = list (zip (x,y,z))
im1 = ax.add_collection3d(art3d.Poly3DCollection([poly],color='c',alpha=0.4))
ax.set_xlabel("p[0]")
ax.set_ylabel("p[1]")
ax.set_zlabel("p[2]")
ax.view_init(elev=30, azim=30)


im2 = ax.plot ([p[0]],[p[1]],[p[2]],"o",color="red")
ims.append (im2)

for _ in range (100):
    p = ch.nextProb (p)
    im2 = ax.plot ([p[0]], [p[1]], [p[2]], "o", color="red")
    ims.append (im2)

ani = anime.ArtistAnimation (fig, ims, interval=50)
ani.save ("anim_3d.mp4",  writer="ffmpeg")

plt.show ()

