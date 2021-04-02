from angles import AngleInterval
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.gca()
# a = AngleInterval(-np.pi/2, 3*np.pi/4)
a = AngleInterval(0, 3*np.pi/4)
a.plot(ax=ax, fc='r')

c = -np.pi
while True:
    b = AngleInterval(c, np.pi/2)
    ax.cla()
    a.plot(ax=ax, fc='r')
    b.plot(ax=ax, fc='c')
    ax.set_title(a.intersects(b, not_equals=True))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    plt.pause(0.1)
    ada=2

    u = a.union(b)
    if u:
        u.plot(ax=ax, fc='y')
        plt.pause(0.1)
        ada=2


    c = c+np.pi/16

# print(a.intersects(b, not_equals=True))
# plt.show()