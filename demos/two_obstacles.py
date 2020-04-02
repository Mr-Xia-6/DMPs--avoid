import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn
import math


# To use the codes in the main folder
import sys
sys.path.insert(0, 'codes/')
sys.path.insert(0, '../codes/')
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=False)

import pdb

from dmp import dmp_cartesian, obstacle_ellipse, point_obstacle

"""
Here we create the trajectory to learn
"""
t_f = 1. * np.pi # final time
t_steps = 10 ** 3 # time steps
t = np.linspace(0, t_f, t_steps)
a_x = 1. / np.pi
b_x = 1.
a_y = 1. / np.pi
b_y = 1.
# x = a_x * t * np.cos(b_x*t)
# y = a_y * t * np.sin(b_y*t)

x = np.linspace(-1,1,1000)
y = -x**2

x_des = np.ndarray([t_steps, 2])
x_des[:, 0] = x
x_des[:, 1] = y
x_des -= x_des[0]

# Learning of the trajectory
dmp = dmp_cartesian.DMPs_cartesian(n_dmps=2, n_bfs=40, K = 1050 * np.ones(2), dt = .01, alpha_s = 3.)
dmp.imitate_path(x_des=x_des)
x_track, dx_track, ddx_track = dmp.rollout()
x_classical = x_track


# Reset state
dmp.reset_state()
x_track = np.zeros((1, dmp.n_dmps))
dx_track = np.zeros((1, dmp.n_dmps))
ddx_track = np.zeros((1, dmp.n_dmps))

dmp.dx_old = np.zeros(dmp.n_dmps)
dmp.ddx_old = np.zeros(dmp.n_dmps)
flag = False
dmp.t = 0
dmp.tol = 5e-02

"""
Volumetric Obstacle
"""
x_track_s = x_track[0]
x_c_1 = -0.5
y_c_1 = 0.63
n = 2
a_1 = .2
b_1 = .1


x_c_2 = 0.15
y_c_2 = 0.4
a_2 = 0.1
b_2 = 0.1
center_1 = np.array([x_c_1, y_c_1])
axis_1 = np.array([a_1, b_1])
center_2 = np.array([x_c_2, y_c_2])
axis_2 = np.array([a_2, b_2])
A = 50.
eta = 1


obst_volume_1 = obstacle_ellipse.Obstacle_Ellipse(n_dim = 2, n = 1, center = center_1, axis = axis_1)
obst_volume_2 = obstacle_ellipse.Obstacle_Ellipse(n_dim = 2, n = 1, center = center_2, axis = axis_2)

while (not flag):
    if (dmp.t == 0):
        dmp.first = True
    else:
        dmp.first = False
    # run and record timestep
    F = (obst_volume_1.compute_forcing_term(x_track_s, A, eta) +
        obst_volume_2.compute_forcing_term(x_track_s, A, eta))

    x_track_s, dx_track_s, ddx_track_s = dmp.step(external_force=F)

    x_track = np.append(x_track, [x_track_s], axis=0)
    dx_track = np.append(dx_track, [dx_track_s],axis=0)
    ddx_track = np.append(ddx_track, [ddx_track_s],axis=0)
    dmp.t += 1
    flag = (dmp.t >= dmp.cs.timesteps) & (np.linalg.norm(x_track_s - dmp.goal) / np.linalg.norm(dmp.goal - dmp.x0) <= dmp.tol)
fig = plt.figure(1)
plt.clf()
plt.figure(1, figsize=(6,6))
plt.plot(x_classical[:,0], x_classical[:, 1], '--b', lw=2, label = 'without obstacle')
e=x_classical[:,0]
f=x_classical[:, 1]
plt.plot(x_track[:,0], x_track[:,1], '-g', lw=2, label = 'with obstacle')

a=x_track[:,0]
b=x_track[:,1]



"""
Point cloud obstacle
"""

dmp.reset_state()
x_track = np.zeros((1, dmp.n_dmps))
dx_track = np.zeros((1, dmp.n_dmps))
ddx_track = np.zeros((1, dmp.n_dmps))

dmp.dx_old = np.zeros(dmp.n_dmps)
dmp.ddx_old = np.zeros(dmp.n_dmps)
flag = False
dmp.t = 0
dmp.tol = 5e-02
# Obstacle definition
num_obst_1 = 50
num_obst_2 = 50
t_1 = np.linspace(0., np.pi * 2., num_obst_1)
t_2 = np.linspace(0., np.pi * 2., num_obst_2)
obst_list_1 = []
obst_list_2 = []
for n in range(num_obst_1):
    obst = point_obstacle.obstacle(x_obst = np.array([x_c_1 + a_1*np.cos(t_1[n]), y_c_1 + b_1*np.sin(t_1[n])]), dx_obst = np.zeros(2))
    obst_list_1.append(obst)

for n in range(num_obst_2):
    obst = point_obstacle.obstacle(x_obst = np.array([x_c_2 + a_2*np.cos(t_2[n]), y_c_2 + b_2*np.sin(t_2[n])]), dx_obst = np.zeros(2))
    obst_list_2.append(obst)

while (not flag):
    if (dmp.t == 0):
        dmp.first = True
    else:
        dmp.first = False
    # run and record timestep
    F_1 = np.zeros([2])
    F_2 = np.zeros([2])
    for n in range(num_obst_1):
        f_n = obst_list_1[n].gen_external_force(dmp.x, dmp.dx, dmp.goal)
        F_1 += f_n
    for n in range(num_obst_2):
        f_n = obst_list_2[n].gen_external_force(dmp.x, dmp.dx, dmp.goal)
        F_2 += f_n
    F = F_1 + F_2
    x_track_s, dx_track_s, ddx_track_s = dmp.step(external_force=F)
    x_track = np.append(x_track, [x_track_s], axis=0)
    dx_track = np.append(dx_track, [dx_track_s],axis=0)
    ddx_track = np.append(ddx_track, [ddx_track_s],axis=0)
    dmp.t += 1
    flag = (dmp.t >= dmp.cs.timesteps) & (np.linalg.norm(x_track_s - dmp.goal) / np.linalg.norm(dmp.goal - dmp.x0) <= dmp.tol)
plt.plot(x_track[:,0], x_track[:,1], color = 'orange', linestyle = '-.', lw=2, label = 'with obstacle')
c=x_track[:,0]
d=x_track[:,1]

xzq=np.zeros((102,2))
for i in range(102):
    h=math.sqrt(((a[i]-e[i]) ** 2) + ((b[i]-f[i])** 2))
    j=math.sqrt(((c[i]-e[i]) ** 2) + ((d[i]-f[i])** 2))
    if (h>j):
        xzq[i][0] = c[i]
        xzq[i][1] = d[i]
    else:
        xzq[i][0] = a[i]
        xzq[i][1] = b[i]
plt.plot(xzq[:,0], xzq[:,1], color = 'k', lw=2)


x_plot_1 = x_c_1 + a_1*np.cos(t_1)
y_plot_1 = y_c_1 + b_1 * np.sin(t_1)
plt.plot (x_plot_1, y_plot_1, ':r', lw=2, label = 'obstacle')
x_plot_2 = x_c_2 + a_2*np.cos(t_2)
y_plot_2 = y_c_2 + b_2 * np.sin(t_2)
plt.plot (x_plot_2, y_plot_2, ':r', lw=2, label = 'obstacle')
plt.xlabel(r'$x_1$',fontsize=14)
plt.ylabel(r'$x_2$',fontsize=14)
plt.axis('equal')
plt.text(dmp.x0[0]-0.05, dmp.x0[1]-0.05, r'$\mathbf{x}_0$', fontsize = 16)
plt.text(dmp.goal[0]+0.01, dmp.goal[1]-0.05, r'$\mathbf{g}$', fontsize = 16)


plt.show()



