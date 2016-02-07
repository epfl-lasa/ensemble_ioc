
"""
Module for utilities
"""

import numpy as np

import scipy
from scipy import interpolate
import scipy.ndimage as spi

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