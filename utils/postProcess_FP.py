# Geubelle research group
# Authors:  Qibang Liu (qibang@illinois.edu)
#           Michael Zakoworotny (mjz7@illinois.edu)
#           Philippe Geubelle (geubelle@illinois.edu)
#           Aditya Kumar (aditya.kumar@ce.gatech.edu)
# 
# Contains a series of useful functions for post-processing data

import os
import numpy as np
from scipy.interpolate import interp1d
from scipy import interpolate as itp
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.animation as animation
import scipy.io as scio



def computeFrontSpeed(x_data, t_data, T_data, alpha_data, save_dir=None, save_suffix="", fid=0):
    """
    Computes the front speed by tracking the location where alpha = 0.5 during the simulation.
    Returns the front speed. If save_dir is supplied, saves the front position vs time plot

    Parameter
    ---------
    x_data, t_data, T_data, alpha_data as returned by FP_solver.solve()

    save_dir - str
        The relative path to which front position vs time plot is saved. If None, image is not saved

    save_suffix - str
        Used to name the save file for front position vs time, which takes the form: "frontPos_{save_suffix}.png"

    fid - int
        The id for plot

    Returns
    -------
    V_fem - float
        The front speed computed from the slope of a linear fit through the front position vs time curve

    """

    # Make save directory
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # Get a list of the front locations by location of alpha = 0.5
    frontLoc_arr = []
    for i, t in enumerate(t_data[:,0]):
        # frontLoc = itp.interp1d(alpha_data[i, :], x_data[i, :],
        #                    kind='nearest', assume_sorted=False, bounds_error=False)
        frontLoc = itp.interp1d(alpha_data[i, :], x_data[i, :], kind='nearest', assume_sorted=False, bounds_error=False)
        frontLoc_arr.append(float(frontLoc(0.5)))

    # Remove data points where nan values are    
    data_arr = np.concatenate((t_data[:,0].reshape(-1, 1), np.array(frontLoc_arr).reshape(-1, 1)), axis=1)
    data_arr = data_arr[~np.isnan(data_arr).any(axis=1), :]

    # Compute front velocity as linear fit through data if possible, else no front found
    if np.any(data_arr):
        V_fem, bbb = np.polyfit(data_arr[:,0], data_arr[:,1], 1)
    else:
        V_fem = 0
        frontLoc_arr = np.zeros_like(t_data)

    # Plot result and print front velocity
    fig = plt.figure(fid)
    ax = fig.add_subplot()
    ax.plot(t_data[:,0], frontLoc_arr, '--', linewidth=2, label='fem')
    ax.set_xlabel('t (s)', fontsize=13)
    ax.set_ylabel(r'$X_{fp}$ (m)', fontsize=13)
    ax.legend()

    if save_dir:
        save_path = os.path.join(save_dir, "frontPos" + ("_"+save_suffix if save_suffix else "") + ".png")
        plt.savefig(save_path, bbox_inches='tight')
    
    print("FP velocity %.3e (mm/s)" % (V_fem*1000))

    return V_fem


def frontAnimation(x_data, t_data, T_data, alpha_data, save_dir, save_suffix="", anim_frames=100, fid=0):
    """
    Save an animation of the temperature and degree of cure fronts

    Parameter
    ---------
    x_data, t_data, T_data, alpha_data as returned by FP_solver.solve()

    save_dir - str
        The relative path to which animation is saved

    save_suffix - str
        Used to name the animation, which takes the form: "frontAnim_{save_suffix}.mov"

    anim_frame - int
        Number of frames in animation

    fid - int
        The id for plot

    """

    # Make save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    
    numFrame = t_data.shape[0] # number of frames
    
    # Create animation and save
    fig = plt.figure(fid)
    ax = fig.add_subplot()
    ax.set_xlabel('x', fontsize=13)
    ax.set_ylabel(r'$\alpha$', fontsize=13)
    ax2 = ax.twinx()
    ax2.set_ylabel(r'$T~^oC$', fontsize=13)
    ims = []
    frames = np.arange(0, numFrame, int(np.ceil((numFrame-1)/anim_frames)))
    frames[-1] = numFrame-1
    for i in frames:
        im1, = ax2.plot(x_data[i, :], T_data[i, :] - 273, '--r', linewidth=2)
        im2, = ax.plot(x_data[i, :], alpha_data[i, :], '--b', linewidth=2) 
        title = ax.text(0.5, 1.05, "time = {:.2f}s".format(t_data[i, 0]),
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax.transAxes, )
        ims.append([im1, im2, title])  # im1,
    ax.legend([im1, im2], [r'$T$', r'$\alpha$'])  # im1,
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False)

    save_path = os.path.join(save_dir, "frontAnim" + ("_"+save_suffix if save_suffix else "") + ".mov")
    ani.save(save_path, writer='ffmpeg')


def saveResults(save_data, save_dir, mat_name):
    """ Saves the supplied dictionary save_data to a matlab data file
    """

    results_file = os.path.join(save_dir, mat_name+".mat")
    scio.savemat(results_file, save_data)

def loadResults(load_dir, mat_name):
    """ Unpacks the supplied .mat file into a dictionary and returns
    """

    results_file = os.path.join(load_dir, mat_name+".mat")
    load_data = scio.loadmat(results_file)
    return load_data