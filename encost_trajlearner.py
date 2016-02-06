"""
Ensemble of cost function based (time-indexed) trajectory learner
"""
from collections import defaultdict
import numpy as np

import ensemble_ioc as eioc

class EnCost_TrajLearner():
    def __init__(self,  n_estimators=20, 
                    max_depth=3, min_samples_split=4, min_samples_leaf=2,
                    random_state=0,
                    em_itrs=0,
                    regularization=0.01,
                    passive_type='inertial',
                    passive_dyn_noise=None,
                    verbose=False,
                    dz=0.01):
        '''
        # most arguments are for the ensemble ioc model
        # the passive_dyn and passive_dyn_ctrl is customized...
        n_estimators        - number of ensembled models
        ...                 - a batch of parameters used for RandomTreesEmbedding, see relevant documents
        em_itrs             - maximum number of EM iterations to take
        regularization      - small positive scalar to prevent singularity of matrix inversion
        passive_type        - the type of the passive behavior, this will decide the construction of trajectory for passive dynamics evaluation
        passive_dyn_noise   - covariance of a Gaussian noise; only applicable when passive_dyn is Gaussian; None for MaxEnt model
                                note this implies a dynamical system with constant input gain. It is extendable to have state dependent
                                input gain then we need covariance for each data point
        verbose             - output training information
        dz                  - the sampling interval of phase variable
        '''
        self.dz = dz
        self.passive_type = passive_type
        self.verbose=verbose

        if passive_type is not None:
            if passive_dyn_noise is None:
                passive_dyn_noise_impl = 1.0
            else:
                passive_dyn_noise_impl = passive_dyn_noise
        else:
            passive_dyn_noise_impl = None


        self.eioc_mdl = eioc.EnsembleIOC(   n_estimators=n_estimators, 
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            random_state=random_state,
                                            em_itrs=em_itrs,
                                            regularization=regularization,
                                            passive_dyn_func=lambda x : x,  #dummy passive dynamics function: trajectory will not change without the control effort
                                            passive_dyn_ctrl=None,          #dummy passive control matrix, NOTE: this value should be changed according to the trajectory length when data is ready...
                                            passive_dyn_noise=passive_dyn_noise_impl,
                                            verbose=verbose 
                                            )
        return

    def train(self, data, phase_scale=None):
        """
        the data is required to be a list of trajectories with equal number of dimensions and sample points
        for each trajectory element, it is a 2D array with rows as state vectors
        """
        self.n_demos = len(data)
        self.n_phases = data[0].shape[0]
        self.n_dofs = data[0].shape[1]
        if phase_scale is None:
            #if we don't care about the scale of phase, we simply assume all the demonstrations are of same time length...
            self.phase_scale = np.ones(self.n_demos)    
        else:
            self.phase_scale = phase_scale

        #remember to update the passive dynamics control matrix: control is applicable at each phase step
        self.eioc_mdl.passive_dyn_ctrl = np.eye(self.n_phases * self.n_dofs)
        #build data
        train_data = self.build_eioc_traj_data_(data)
        #fit with the eioc model
        if self.verbose:
            print 'fitting...'
        #<hyin/Feb-6th-2016> remember now the flattened trajectories are now suffixed with the phase scale
        indices, leaf_idx_dict, passive_likelihood_dict = self.eioc_mdl.fit(train_data[:, 0:-1])
        #extract desired information...
        if self.verbose:
            print 'populating learner parameters...'
        self.populate_learner_parms_(train_data, indices, leaf_idx_dict, passive_likelihood_dict)

        return

    def build_eioc_traj_data_(self, data):
        """
        build eioc format compatible data:
        if the passive type is None, then use MaxEnt formulation
        otherwise, need to construct the passive state...
        """
        #<hyin/Feb-6th-2016> add extra phase scale for each trajectory for potential encodement of time horizon
        if self.passive_type is None:
            #MaxEnt formulation: just flatten each DOF
            res_data = np.array([np.concatenate([traj.T.flatten(), [ps]]) for traj, ps in zip(data, self.phase_scale)])
        else:
            passive_data = [self.construct_passive_state_(traj, self.passive_type).T.flatten() for traj in data]
            #now concatenate them...
            res_data = np.array([np.concatenate([passive_traj, traj.T.flatten(), [ps]]) for passive_traj, traj, ps in zip(passive_data, data, self.phase_scale)])

        return res_data

    def construct_passive_state_(self, traj, passive_type='inertial'):
        """
        construct a virtual state for the trajectory to evaluate dynamics
        here we take the trajectory as a state denoted by a long vector
        the passive behavior of the trajectory from the initial point could be:
        stationary: always stay at the initial point
        inertial:   has a constant velocity as the  
        """
        if passive_type == 'stationary':
            return np.array([traj[0] for s in traj])
        elif passive_type == 'inertial':
            #for each dof, do linear intepolation 
            return np.array([np.linspace(traj[0, dim], traj[-1, dim], traj.shape[0]) for dim in range(traj.shape[1])]).T
        elif passive_type == 'damping':
            #TODO: a damped passive behavior 
            pass

        return

    def populate_learner_parms_(self, train_data, indices, leaf_idx_dict, passive_likelihood_dict):
        """
        this is the function to construct trajectory favored format from the eioc results...
        """
        self.mean_trajs = []
        self.tracking_mats = []
        self.cost_consts = []
        self.mode_weights = []

        # print len(self.eioc_mdl.estimators_full_['means'])
        # print len(self.eioc_mdl.estimators_full_['covars'])
        # print len(self.eioc_mdl.estimators_full_['beta'])
        # print len(self.eioc_mdl.estimators_full_['weights'])
        # print zip(    self.eioc_mdl.estimators_full_['means'],
                        # self.eioc_mdl.estimators_full_['covars'],
                        # self.eioc_mdl.estimators_full_['beta'],
                        # self.eioc_mdl.estimators_full_['weights'])

        self.mode_phase_scales = []
        self.mode_phase_scales_var = []
        #assign the phase scale to each partition...
        phase_scale_dict = defaultdict(list)
        for d, data_partition_indices in zip(train_data, indices):
            #data_partition_idx is a list of leaf index for each estimator...
            for e_idx, l_idx in enumerate(data_partition_indices):
                phase_scale_dict[e_idx, l_idx].append(d[-1])    #push back the time scale for this data trajectory

        #<hyin/Feb-6th-2016> construct the averaged phase scale according to the passive likelihood...
        for e_idx in range(self.eioc_mdl.n_estimators):
            for l_idx in leaf_idx_dict[e_idx]:
                self.mode_phase_scales.append(np.sum(np.array(passive_likelihood_dict[e_idx, l_idx]) * np.array(phase_scale_dict[e_idx, l_idx])))
                self.mode_phase_scales_var.append(np.array(np.sum(np.array(passive_likelihood_dict[e_idx, l_idx]) * phase_scale_dict[e_idx, l_idx]) - self.mode_phase_scales[-1])**2)

        for mode_idx, (mean_traj, traj_covar, traj_cost_const, weights) in enumerate(zip(    self.eioc_mdl.estimators_full_['means'],
                                                                                                self.eioc_mdl.estimators_full_['covars'],
                                                                                                self.eioc_mdl.estimators_full_['beta'],
                                                                                                self.eioc_mdl.estimators_full_['weights'])):
            #restore the trajectory by stack the vector
            # #NOTE here we must deal with different passive type, non Max-Ent will give augmented data...
            # if self.passive_type is not None:
            #     self.mean_trajs.append(np.reshape(mean_traj[len(mean_traj)/2:], (self.n_dofs, self.n_phases)).T)
            #     #extract correlated terms to construct the variance for each phase step
            #     tmp_covar_mats = [traj_covar[np.array([traj_covar.shape[0]/2 + i+np.arange(self.n_dofs)*self.n_phases]).T.tolist(), traj_covar.shape[1]/2+i+np.arange(self.n_dofs)*self.n_phases] for i in range(self.n_phases)]
            # else:
            self.mean_trajs.append(np.reshape(mean_traj, (self.n_dofs, self.n_phases)).T)
            #extract correlated terms to construct the variance for each phase step
            tmp_covar_mats = [traj_covar[np.array([i+np.arange(self.n_dofs)*self.n_phases]).T.tolist(), i+np.arange(self.n_dofs)*self.n_phases] for i in range(self.n_phases)]
            self.tracking_mats.append([np.linalg.pinv(mat + np.eye(mat.shape[0])*0.01) for mat in tmp_covar_mats])
            #for the constant term
            self.cost_consts.append([-0.5*np.log(eioc.pseudo_determinant(mat)) + 0.5*np.log(np.pi*2)*self.n_dofs for mat in self.tracking_mats[-1]])
            # print weights
            self.mode_weights.append(weights)
        assert(len(self.mode_phase_scales) == len(self.mean_trajs))
        assert(len(self.mode_phase_scales_var)) == len(self.mean_trajs)
        return

"""
Unit test:
Generate a few trajectories with two modes, and use the traj learner to encode them...
"""
import pyrbf_funcapprox as pyrbf_fa

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def EnCost_TrajLearner_Test():
    #prepare data
    start_pnt = np.array([0, 0, 0])
    end_pnt1 = np.array([1, 1, 1])
    end_pnt2 = np.array([-1, -1, 1])

    n_phases = 100
    z = np.linspace(0, 1.0, n_phases)

    line1 = [np.linspace(start_pnt[dof], end_pnt1[dof], n_phases) for dof in range(len(start_pnt))]
    line2 = [np.linspace(start_pnt[dof], end_pnt2[dof], n_phases) for dof in range(len(start_pnt))]

    n_samples = 10
    gen_data_1 = []
    gen_data_2 = []
    for dof in range(len(start_pnt)):
        line1_fa = pyrbf_fa.PyRBF_FunctionApproximator(rbf_type='sigmoid', K=10, normalize=True)
        line2_fa = pyrbf_fa.PyRBF_FunctionApproximator(rbf_type='sigmoid', K=10, normalize=True)
        #fix initial points...
        line1_fa.set_linear_equ_constraints([z[0]], [line1[dof][0]])
        line2_fa.set_linear_equ_constraints([z[0]], [line2[dof][0]])
        #fit
        line1_fa.fit(z, line1[dof], replace_theta=True)
        line2_fa.fit(z, line2[dof], replace_theta=True)
        #generate samples...
        dof_samples_lin1 = line1_fa.gaussian_sampling(noise=0.05, n_samples=n_samples)
        dof_samples_lin2 = line2_fa.gaussian_sampling(noise=0.05, n_samples=n_samples)     
        gen_data_1.append(np.array([line1_fa.evaluate(z=z, theta=s) for s in dof_samples_lin1]))
        gen_data_2.append(np.array([line2_fa.evaluate(z=z, theta=s) for s in dof_samples_lin2]))
    #reconstruct data to have a suitable format
    data_1 = [np.array([gen_data_1[dof_idx][sample_idx, :] for dof_idx in range(len(start_pnt))]).T for sample_idx in range(n_samples)]
    data_2 = [np.array([gen_data_2[dof_idx][sample_idx, :] for dof_idx in range(len(start_pnt))]).T for sample_idx in range(n_samples)]
    #concatenate them...
    data = data_1 + data_2
    #show data
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.hold(True)
    for traj in data:
        ax.plot(xs=traj[:, 0], ys=traj[:, 1], zs=traj[:, 2], linestyle='--', color='b', alpha=0.6)
    plt.draw()

    #try the trajectory learner
    traj_learner = EnCost_TrajLearner(n_estimators=5, passive_type=None, verbose=True)
    traj_learner.train(data=data)

    #see the mean trajs
    print traj_learner.mode_weights
    print np.exp(-np.mean(np.array(traj_learner.cost_consts), axis=1))
    print traj_learner.mode_phase_scales
    transp = traj_learner.mode_weights * np.exp(-np.mean(np.array(traj_learner.cost_consts), axis=1)) / np.amax(np.array(traj_learner.mode_weights) * np.exp(-np.mean(np.array(traj_learner.cost_consts), axis=1)))
    for mode_idx, mean_traj in enumerate(traj_learner.mean_trajs):
        ax.plot(xs=mean_traj[:, 0], ys=mean_traj[:, 1], zs=mean_traj[:, 2], color='k', linewidth=2.5, alpha=transp[mode_idx])
    plt.draw()
    return


