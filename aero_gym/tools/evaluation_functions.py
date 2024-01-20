import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation
import numpy as np
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

"""
Run an episode of environment `env`, performing the actions in `alpha_ddot_prescribed`, if defined. 
"""
def evaluate(env, alpha_ddot_prescribed=None):
    env_with_stats = RecordEpisodeStatistics(env)
    render_list = []
    obs, info = env_with_stats.reset()
    i = 0
    while 1:
        render_list.append(env_with_stats.render())
        if alpha_ddot_prescribed is None:
            action = [0.0]
        else:
            action = [alpha_ddot_prescribed[i]]
        
        obs, _, terminated, truncated, info = env_with_stats.step(action)
        if terminated or truncated:
            break
        i += 1
    return obs, info, render_list

"""
Plot the environment states and statistics in `info_dict`, generated with `evaluate`.
"""
def plotfile(info_dict, axarr=None, label=None):
    if axarr is None:
        fig, axarr = plt.subplots(ncols=3, nrows=2, figsize=(17,8))
    else:
        fig = axarr[0,0].figure

    for ax in axarr.flatten():
        ax.minorticks_on()
        ax.grid(which='both',axis='y')
        ax.set_xlabel('time')

    axarr[0,0].set_ylabel('total lift')
    axarr[0,0].plot(info_dict["t_hist"], info_dict["fy_hist"], label=label, marker='o', markersize=2, linewidth=1)
    axarr[0,1].set_ylabel('h dot')
    axarr[0,1].plot(info_dict["t_hist"], info_dict["h_dot_hist"], marker='o', markersize=2, linewidth=1)
    axarr[0,2].set_ylabel('h ddot')
    axarr[0,2].step(info_dict["t_hist"], info_dict["h_ddot_hist"], where='post', marker='o', markersize=2, linewidth=1)
    axarr[1,0].set_ylabel('alpha [deg]')
    axarr[1,0].plot(info_dict["t_hist"], info_dict["alpha_hist"] * 180 / np.pi, marker='o', markersize=2, linewidth=1)
    axarr[1,1].set_ylabel('alpha dot')
    axarr[1,1].plot(info_dict["t_hist"], info_dict["alpha_dot_hist"], marker='o', markersize=2, linewidth=1)
    axarr[1,2].set_ylabel('alpha ddot')
    axarr[1,2].step(info_dict["t_hist"], info_dict["alpha_ddot_hist"], where='post', marker='o', markersize=2, linewidth=1)

    if label is not None:
        axarr[0,0].legend()

    return fig, axarr

"""
Animate grayscale renders of vorticity in `renderlist` using the episode statistics in `render_list`. The rotation of the airfoil is performed about the pixels provided in `pivot_idx`.
"""
def animaterender(info_dict, render_list, pivot_idx, animate_every=10):
    fig, ax = plt.subplots();
    im = ax.imshow(render_list[0][:,:,0]);

    def animate(i):
        im.set_array(render_list[i][:,:,0])
        im.set_transform(Affine2D().rotate_deg_around(pivot_idx[0], pivot_idx[1], info_dict["alpha_hist"][i] * 180 / np.pi) + plt.gca().transData)

    anim = FuncAnimation(fig, animate, frames=range(0, len(render_list), animate_every))

    return anim

"""
Animate grid renders of vorticity in `renderlist` using the episode statistics in `render_list` and physical grid coordinates in `xg` and `yg`. The rotation of the airfoil is performed about the coordinates provided in `pivot_idx`.
"""
def animaterender_contour(info_dict, xg, yg, render_list, pivot_idx, levels, vmin=-20, vmax=20, quiver_pivot=[0,0], subplot_kw={}, fig_kw={}, show_inflow=False, quiver_kw={}, animate_every=10, interval=200, alpha_init=0):
    fig, ax = plt.subplots(subplot_kw=subplot_kw, **fig_kw); 
    ax.axis('scaled')

    def animate(i):
        ax.clear()
        cont = plt.contourf(xg[3:-3], yg[3:-3], render_list[i][3:-3,3:-3], levels, vmin=vmin, vmax=vmax, cmap='bwr');
        for c in cont.collections:
            c.set_transform(Affine2D().rotate_deg_around(pivot_idx[0], pivot_idx[1], -info_dict["alpha_hist"][i] * 180 / np.pi) + plt.gca().transData)
        plate, = ax.plot([-0.5,0.5],[0,0],c='black',lw=3)
        plate.set_transform(Affine2D().rotate_deg_around(pivot_idx[0], pivot_idx[1], (-alpha_init - info_dict["alpha_hist"][i]) * 180 / np.pi) + plt.gca().transData)

        if show_inflow:
            ax.quiver(*quiver_pivot, 1.0, -info_dict["h_dot_hist"][i], **quiver_kw)

    anim = FuncAnimation(fig, animate, frames=range(0, len(render_list), animate_every), interval=interval)

    return anim
