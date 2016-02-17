
"""
Module for utilities
"""

import numpy as np

import scipy
from scipy import interpolate
import scipy.ndimage as spi

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def load_data_from_txt_file(fname, position_field_idx = [4, 5, 6], skip_header_ratio=0.1, skip_rear_ratio=0.1):
    """
    load the trajectory data from the txt file which contains a stream of ros type PoseStamped data type
    the function concerning the field of position.x:y:z
    skip_header_ratio/skip_rear_ratio decide the proportional data to omit because the sensor reading for these period of time
    might be meaningless
    """
    full_data = np.genfromtxt(fname, delimiter=',', skip_header=1, converters={3: lambda s: float(s or 0)})
    len_data = len(full_data)
    data_clip = full_data[int(len_data*skip_header_ratio):int(len_data*(1-skip_rear_ratio)), position_field_idx]
    return data_clip

def display_spatial_trajectory_data(data):
    """
    data would be a list of spatial trajectories with row as a 3-dimension state variable...
    """
    """
    <hyin/Feb-9th-2016> the 3d plotting of matplotlib hasn't resolve the equal axis issue
    use a workaround by forcing the axis limit
    """
    max_lst = []
    min_lst = []
    mean_lst = []
    for d in data:
        max_lst.append(np.amax(d, axis=0))
        min_lst.append(np.amin(d, axis=0))
        mean_lst.append(np.mean(d, axis=0))

    #figure out the max & min for each dimension throughout the whole data...
    max_coord = np.amax(max_lst, axis=0)
    min_coord = np.amin(min_lst, axis=0)
    mean_coord = np.mean(mean_lst, axis=0)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.hold(True)
    
    for d in data:
        ax.plot(xs=d[:, 0], ys=d[:, 1], zs=d[:, 2], linestyle='--', color='b', alpha=0.6)

    ax_lim = [mean_coord - np.max((max_coord-min_coord))/2.0, mean_coord + np.max((max_coord-min_coord))/2.0]
    ax.set_xlim(ax_lim[0][0], ax_lim[1][0])
    ax.set_ylim(ax_lim[0][1], ax_lim[1][1])
    ax.set_zlim(ax_lim[0][2], ax_lim[1][2])

    plt.draw()
    
    return

def display_spatial_trajectory_pos_and_vel_data(data):
    """
    data would be a list of spatial trajectories with row as a 6-dimension state variable...
    with the first 3 dimensions as position and the remained 3 dimensions as velocity
    """
    """
    <hyin/Feb-9th-2016> the 3d plotting of matplotlib hasn't resolve the equal axis issue
    use a workaround by forcing the axis limit
    """
    max_lst = []
    min_lst = []
    mean_lst = []
    for d in data:
        max_lst.append(np.amax(d, axis=0))
        min_lst.append(np.amin(d, axis=0))
        mean_lst.append(np.mean(d, axis=0))

    #figure out the max & min for each dimension throughout the whole data...
    max_coord = np.amax(max_lst, axis=0)
    min_coord = np.amin(min_lst, axis=0)
    mean_coord = np.mean(mean_lst, axis=0)

    plt.ion()
    fig = plt.figure()
    ax_pos = fig.add_subplot(121, projection='3d')
    ax_vel = fig.add_subplot(122, projection='3d')
    ax_pos.hold(True)
    ax_vel.hold(True)
    
    for d in data:
        ax_pos.plot(xs=d[:, 0], ys=d[:, 1], zs=d[:, 2], linestyle='--', color='b', alpha=0.6)
        ax_vel.plot(xs=d[:, 3], ys=d[:, 4], zs=d[:, 5], linestyle='--', color='b', alpha=0.6)

    ax_pos_lim = [mean_coord[:3] - np.max((max_coord-min_coord)[:3])/2.0, mean_coord[:3] + np.max((max_coord-min_coord)[:3])/2.0]
    ax_vel_lim = [mean_coord[3:] - np.max((max_coord-min_coord)[3:])/2.0, mean_coord[3:] + np.max((max_coord-min_coord)[3:])/2.0]

    ax_pos.set_xlim(ax_pos_lim[0][0], ax_pos_lim[1][0])
    ax_pos.set_ylim(ax_pos_lim[0][1], ax_pos_lim[1][1])
    ax_pos.set_zlim(ax_pos_lim[0][2], ax_pos_lim[1][2])

    ax_vel.set_xlim(ax_vel_lim[0][0], ax_vel_lim[1][0])
    ax_vel.set_ylim(ax_vel_lim[0][1], ax_vel_lim[1][1])
    ax_vel.set_zlim(ax_vel_lim[0][2], ax_vel_lim[1][2])

    plt.draw()
    
    return

def smooth_data(data, size=3):
    """
    a function to smooth trajectories
    """
    smoothed_trajs = []
    for traj_idx, traj in enumerate(data):
        time_len = len(traj)
        if time_len > 3:
            if len(traj.shape) == 1:
                """
                mono-dimension trajectory, row as the entire trajectory...
                """
                filtered_traj = spi.gaussian_filter(traj, size)
            else:
                """
                multi-dimensional trajectory, row as the state variable...
                """
                filtered_traj = np.array([spi.gaussian_filter(traj_dof, size) for traj_dof in traj.T]).T
            smoothed_trajs.append(filtered_traj)
        else:
            print 'The {0} trajectory is too short, ignored...'.format(traj_idx)

    return smoothed_trajs

def interp_data_fixed_num_phases(data, dt=0.01, num=100):
    """
    a function to align multi-dimensional trajectories by resampling
    with same number of phases point
    also return the scale between phase sample and original equal-time interval (dt) sample
    the data consists of a list of trajectories with same dimension but not necessarily
    same time horizon length
    the returned aligned data is the representation of the original trajectories with same
    amount of phase points...
    """
    aligned_trajs = []
    phase_scale = []

    for traj_idx, traj in enumerate(data):
        time_len = len(traj)
        t = np.linspace(0, 1, time_len)
        t_spl = np.linspace(0, 1, num)
        if time_len > 3:
            #we need at least 3 data points for each trajectory...
            if len(traj.shape) == 1:
                """
                mono-dimension trajectory, row as the entire trajectory...
                """
                spl = interpolate.splrep(t, traj)
                traj_interp = interpolate.splev(t_spl, spl, der=0)

                aligned_trajs.append(traj_interp)
            else:
                """
                multi-dimensional trajectory, row as the state variable...
                """
                tmp_aligned_traj = []
                for traj_dof in traj.T:
                    spl = interpolate.splrep(t, traj_dof)
                    traj_interp = interpolate.splev(t_spl, spl, der=0)
                    tmp_aligned_traj.append(traj_interp)
                aligned_trajs.append(np.array(tmp_aligned_traj).T)
            phase_scale.append(float(time_len)*dt/num)
        else:
            print 'The {0} trajectory is too short, ignored...'.format(traj_idx)

    return aligned_trajs, phase_scale

def expand_traj_dim_with_derivative(data, dt=0.01):
    augmented_trajs = []
    for traj in data:
        time_len = len(traj)
        t = np.linspace(0, time_len*dt, time_len)
        if time_len > 3:
            if len(traj.shape) == 1:
                """
                mono-dimension trajectory, row as the entire trajectory...
                """
                spl = interpolate.splrep(t, traj)
                traj_der = interpolate.splev(t, spl, der=1)
                tmp_augmented_traj = np.array([traj, traj_der]).T
            else:
                """
                multi-dimensional trajectory, row as the state variable...
                """
                tmp_traj_der = []
                for traj_dof in traj.T:
                    spl_dof = interpolate.splrep(t, traj_dof)
                    traj_dof_der = interpolate.splev(t, spl_dof, der=1)
                    tmp_traj_der.append(traj_dof_der)
                tmp_augmented_traj = np.vstack([traj.T, np.array(tmp_traj_der)]).T

            augmented_trajs.append(tmp_augmented_traj)

    return augmented_trajs

def extract_traj_dim(data, dims):
    """
    extract the specific dimensions of data
    """
    res_data = [d[:, dims] for d in data]
    

    return res_data

def augment_traj_dim(data, aug_data):
    res_data = [np.vstack([d.T, aug_d.T]).T for d, aug_d in zip(data, aug_data)]
    return res_data