import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation
import math
import numpy as np

"""
Run an episode of environment `env`, performing the actions in `alpha_ddot_prescribed`, if defined, and write the environment states and statistics to `filename`. 
"""
def evaluate(env, filename, alpha_ddot_prescribed=None):
    render_list = []
    writefile = open(filename, "w")
    obs, info = env.reset()
    episode_return = 0
    i = 0
    while 1:
        render_list.append(env.render())
        if alpha_ddot_prescribed is None:
            action = [0.0]
        else:
            action = [alpha_ddot_prescribed[i]]
        
        writestr = ("{:5d}" + "{:11.5f}" + "{:11.3e} " * 4).format(
            info["time_step"], 
            info["t"], 
            info["unscaled h_dot"], 
            info["unscaled alpha"], 
            info["unscaled alpha_dot"], 
            info["unscaled h_ddot"])
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        writestr += ("{:11.3e} " * 4).format(info["unscaled previous alpha_ddot"], info["unscaled previous fy"], reward, episode_return)
        writefile.write(writestr + "\n")
        i+=1
        if terminated or truncated:
            break
    writefile.close()
    return obs, info, render_list

"""
Plot the environment states and statistics in `filename`, generated with `evaluate`.
"""
def plotfile(filename, axarr=None, label=None, a=0):
    if axarr is None:
        fig, axarr = plt.subplots(ncols=3, nrows=3, figsize=(17,12))
    else:
        fig = axarr[0,0].figure
    axarr[0,0].set_ylabel('total lift')
    axarr[0,0].set_xlabel('time')
    axarr[0,1].set_ylabel('h dot')
    axarr[0,1].set_xlabel('time')
    axarr[0,2].set_ylabel('h ddot')
    axarr[0,2].set_xlabel('time')
    axarr[1,0].set_ylabel('alpha')
    axarr[1,0].set_xlabel('time')
    axarr[1,1].set_ylabel('alpha dot')
    axarr[1,1].set_xlabel('time')
    axarr[1,2].set_ylabel('alpha ddot')
    axarr[1,2].set_xlabel('time')
    axarr[2,0].set_ylabel('reward')
    axarr[2,0].set_xlabel('time step')
    axarr[2,1].set_ylabel('return (sum of rewards)')
    axarr[2,1].set_xlabel('time step')
    axarr[2,2].set_ylabel('time')
    axarr[2,2].set_xlabel('time step')

    with open(filename, "r") as readFile:
        textstr = readFile.readlines()
        all_lines = [line.split() for line in textstr]
        timestep_hist = [float(x[0]) for x in all_lines]
        t_hist = [float(x[1]) for x in all_lines]
        h_dot_list = [float(x[2]) for x in all_lines]
        alpha_list = [float(x[3]) for x in all_lines]
        alpha_dot_list = [float(x[4]) for x in all_lines]
        h_ddot_list = [float(x[5]) for x in all_lines]
        alpha_ddot_list = [float(x[6]) for x in all_lines]
        fy_hist = [float(x[7]) for x in all_lines]
        am_list = [math.pi / 4 * (-h_ddot_list[i] - a * alpha_ddot_list[i] + alpha_dot_list[i]) for i in range(len(h_ddot_list))]
        reward_list = [float(x[8]) for x in all_lines]
        episode_return_list = [float(x[9]) for x in all_lines]
        readFile.close()
            
    axarr[0,0].plot(t_hist,fy_hist, label=label, marker='o', markersize=2, linewidth=1)
    axarr[0,1].plot(t_hist,h_dot_list, marker='o', markersize=2, linewidth=1)
    axarr[0,2].plot(t_hist,h_ddot_list, marker='o', markersize=2, linewidth=1)
    axarr[1,0].plot(t_hist,alpha_list, marker='o', markersize=2, linewidth=1)
    axarr[1,1].plot(t_hist,alpha_dot_list, marker='o', markersize=2, linewidth=1)
    axarr[1,2].plot(t_hist,alpha_ddot_list, marker='o', markersize=2, linewidth=1)
    axarr[2,0].plot(timestep_hist,reward_list, marker='o', markersize=2, linewidth=1)
    axarr[2,1].plot(timestep_hist,episode_return_list, marker='o', markersize=2, linewidth=1)
    axarr[2,2].plot(timestep_hist[:-1],np.diff(t_hist), marker='o', markersize=2, linewidth=1)

    if label is not None:
        axarr[0,0].legend()

    for ax in axarr.flatten():
        ax.minorticks_on()
        ax.grid(which='both',axis='y')
    return fig, axarr

"""
Animate grayscale renders of vorticity in `renderlist` using the episode statistics in `render_list`. The rotation of the airfoil is performed about the pixels provided in `pivot_idx`.
"""
def animaterender(filename, render_list, pivot_idx, animate_every=10):
    with open(filename, "r") as readFile:
        textstr = readFile.readlines()
        all_lines = [line.split() for line in textstr]
        alpha_list = [float(x[3]) for x in all_lines]
        readFile.close()
        
    fig, ax = plt.subplots();
    im = ax.imshow(render_list[0][:,:,0]);

    def animate(i):
        im.set_array(render_list[i][:,:,0])
        im.set_transform(Affine2D().rotate_deg_around(*pivot_idx, alpha_list[i] * 180 / np.pi) + plt.gca().transData)

    anim = FuncAnimation(fig, animate, frames=range(0, len(render_list), animate_every))

    return anim

"""
Animate grid renders of vorticity in `renderlist` using the episode statistics in `render_list` and physical grid coordinates in `xg` and `yg`. The rotation of the airfoil is performed about the coordinates provided in `pivot_idx`.
"""
def animaterender_contour(filename, xg, yg, render_list, pivot_idx, levels, vmin=-20, vmax=20, subplot_kw={}, fig_kw={}, show_inflow=False, quiver_kw=None, animate_every=10, interval=200, alpha_init=0):
    with open(filename, "r") as readFile:
        textstr = readFile.readlines()
        all_lines = [line.split() for line in textstr]
        alpha_list = [float(x[3]) for x in all_lines]
        h_dot_list = [float(x[2]) for x in all_lines]
        readFile.close()
        
    fig, ax = plt.subplots(subplot_kw=subplot_kw, **fig_kw); 
    ax.axis('scaled')

    def animate(i):
        ax.clear()
        cont = plt.contourf(xg[3:-3], yg[3:-3], render_list[i][3:-3,3:-3], levels, vmin=vmin, vmax=vmax, cmap='bwr');
        for c in cont.collections:
            c.set_transform(Affine2D().rotate_deg_around(*pivot_idx, -alpha_list[i] * 180 / np.pi) + plt.gca().transData)
        plate, = ax.plot([-0.5,0.5],[0,0],c='black',lw=3)
        plate.set_transform(Affine2D().rotate_deg_around(*pivot_idx, (-alpha_init - alpha_list[i]) * 180 / np.pi) + plt.gca().transData)

        if show_inflow:
            ax.quiver(*pivot_idx, 1.0, -h_dot_list[i], **quiver_kw)

    anim = FuncAnimation(fig, animate, frames=range(0, len(render_list), animate_every), interval=interval)

    return anim
