"""
Ensemble of cost function based (time-indexed) trajectory learner
"""
import cPickle as cp
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
                self.mode_phase_scales.append(self.n_phases*np.sum(np.array(passive_likelihood_dict[e_idx, l_idx]) * np.array(phase_scale_dict[e_idx, l_idx])))
                self.mode_phase_scales_var.append(self.n_phases**2 * np.array(np.sum(np.array(passive_likelihood_dict[e_idx, l_idx]) * phase_scale_dict[e_idx, l_idx]) - self.mode_phase_scales[-1])**2)

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

    def unpickle(self, fname):
        data =  cp.load(open(fname, 'rb'))
        if 'mean_trajs' in data:
            self.mean_trajs = data['mean_trajs']
        if 'tracking_mats' in data:
            self.tracking_mats = data['tracking_mats']
        if 'cost_consts' in data:
            self.cost_consts = data['cost_consts']
        if 'mode_weights' in data:
            self.mode_weights = data['mode_weights']
        if 'mode_phase_scales' in data:
            self.mode_phase_scales = data['mode_phase_scales']
        if 'mode_phase_scales_var' in data:
            self.mode_phase_scales_var = data['mode_phase_scales_var']

        return

    def pickle(self, fname):
        f = file(fname, 'wb')
        data = defaultdict(list)
        if hasattr(self, 'mean_trajs'):
            data['mean_trajs'] = self.mean_trajs
        if hasattr(self, 'tracking_mats'):
            data['tracking_mats'] = self.tracking_mats
        if hasattr(self, 'cost_consts'):
            data['cost_consts'] = self.cost_consts
        if hasattr(self, 'mode_weights'):
            data['mode_weights'] = self.mode_weights
        if hasattr(self, 'mode_phase_scales'):
            data['mode_phase_scales'] = self.mode_phase_scales
        if hasattr(self, 'mode_phase_scales_var'):
            data['mode_phase_scales_var'] = self.mode_phase_scales_var

        cp.dump(data, f)
        f.close()
        return

class EnCost_ImpController:
    def __init__(self, 
        learner,         
        update_dt = 0.05, obs_tl=False):
        self.learner = learner
        self.update_dt = update_dt

        self.model=None
        self.n_modes=None
        self.trans_mat=None
        self.mode_belief=None
        self.ref_trajs = []
        self.track_weights = []
        self.feedback_gains = []
        self.mode_phase_scales = []
        self.mode_phase_scales_var = []
        self.curr_state = None
        self.realized_state_storage = None
        self.obs_tl=obs_tl

        return

    def initialize_controller(self, mode=None, eigen_prob=0.9):
        if self.learner is None:
            print 'The ensemble cost model is not initialized yet.'
            return
        else:
            print 'Reinitialize the controller...'
            print 'The original reference command and feedback gain will be reset'
            self.ref_trajs = []
            self.track_weights = []
            self.feedback_gains = []
            self.cost_consts = []
            self.mode_weights = []
            self.mode_phase_scales = []
            self.mode_phase_scales_var = []

        self.ref_trajs = self.learner.mean_trajs
        self.track_weights = self.learner.tracking_mats
        self.cost_consts = self.learner.cost_consts
        self.mode_weights = self.learner.mode_weights
        self.mode_phase_scales = self.learner.mode_phase_scales
        self.mode_phase_scales_var = self.learner.mode_phase_scales_var

        self.n_modes        = len(self.ref_trajs)
        print '{0} modes of ensemble models are loaded.'.format(self.n_modes)
        # self.mode_belief    = np.zeros(self.n_modes)
        self.mode_belief    = np.ones(self.n_modes) * (1.0-eigen_prob)/(self.n_modes-1)
        if mode is None:
            b_rand_mode = True
        elif mode > self.n_modes or mode < 0:
            b_rand_mode = True
        else:
            b_rand_mode = False

        if b_rand_mode:
            print 'use a random mode.'
            init_mode = np.random.choice(self.n_modes)
        else:
            print 'use specified mode'
            init_mode = mode

        # self.mode_belief[init_mode] = 1.0
        self.curr_mode = init_mode
        self.mode_belief[init_mode] = eigen_prob

        #for transition matrix - the belief propagation of mode
        self.trans_mat = np.ones((self.n_modes, self.n_modes)) * (1.0-eigen_prob)/(self.n_modes-1)
        for mode_idx in range(self.n_modes):
            self.trans_mat[mode_idx, mode_idx] = eigen_prob

        #initialize the sensory perception
        self.set_sensor_feedback(self.ref_trajs[init_mode][0, :])
        #note that the phase index is actually related to the trajectory length...
        self.curr_idx = np.zeros(self.n_modes)
        return

    def set_sensor_feedback(self, obs):
        """
        the observation should be concrete sensed spatial coordinate
        which might subject to both command as well as external perturbation
        """
        self.curr_state = obs
        return

    def get_interp_fdfwd_and_fdbck_(self, mode, p):
        #for feedforward trajector command (reference regulation), use linear intepolation
        #use zero order hold for tracking mat...
        #note here we use the tracking mat as a surrogate of the feedback gain, that's because
        #we dont have explicit conrol effort regulation
        floor_idx = int(np.floor(p))
        ceil_idx = int(np.ceil(p))
        if ceil_idx >= len(self.ref_trajs[mode]):
            #reach the end...
            return self.ref_trajs[mode][-1, :], self.track_weights[mode][-1]
        else:
            ref_traj_interp = (self.ref_trajs[mode][ceil_idx, :] - self.ref_trajs[mode][floor_idx, :]) * (p - floor_idx) + self.ref_trajs[mode][floor_idx, :]
            if p - floor_idx > 0.5:
                tracking_mats_zero_order = self.track_weights[mode][ceil_idx]
            else:
                tracking_mats_zero_order = self.track_weights[mode][floor_idx]

            return ref_traj_interp, tracking_mats_zero_order

    def evaluate_cost_to_go_(self, mode, obs, t_idx=None):
        """
        evaluate the cost-to-go for given observation and mode, if t_idx is not given use the current one
        """
        if t_idx is None:
            idx = self.curr_idx[mode]
        else:
            idx = t_idx

        #<hyin/Feb-7th-2016> note the idx might not neessarily to be an integer now
        #in fact, it's a float phase variable indicating the process evolution
        #so we need some intepolation to deal with the reference trajectory and tracking weights...
        ref_pnt, weight_mat = self.get_interp_fdfwd_and_fdbck_(mode, idx)

        # ref_pnt = self.ref_trajs[mode][idx, :]
        # weight_mat = self.track_weights[mode][idx]

        # cost = (obs - ref_pnt).dot(weight_mat.dot(obs-ref_pnt))
        cost = (obs[3:] - ref_pnt[3:]).dot((obs[3:] - ref_pnt[3:])) * 5.0
        # cost = (obs - ref_pnt).dot(obs - ref_pnt) * 5.0
        return cost
    def evaluate_cost_to_go_ti_(self, mode, obs):
        """
        evaluate the cost-to-go for given observation and mode, in a time invariant manner...
        in fact we take the evaluation of cost-to-go along the mode trajectory and choose the most_probable_mode_idx
        probable one as the trajectory cost-to-go, remember to also decide the phase index (current task completion)
        """
        cost_to_go_traj = [(obs - ref_pnt).dot(weight_mat.dot(obs-ref_pnt)) for ref_pnt, weight_mat in zip(self.ref_trajs[mode], self.track_weights[mode])]
        #take the smallest one
        min_idx = np.argmin(cost_to_go_traj)
        return min_idx, cost_to_go_traj[min_idx]

    def update_mode_state(self):
        """
        update the internal belief about the mode of ensemble models
        P(s'|o', o'', ..., o^0) \prop P(s', o', o, ..., o^0) = P(o'|s')P(s'|o, ..., o^0)
        = P(o'|s') \int_{s} P(s'|s) P(s|o, ..., o^0)
        
        P(s'|s, o') \prop P(s', s, o') = P(o'|s')P(s'|s)P(s)
        """
        #propagate of mode belief P(s'|s)
        tmp_mode_belief = self.trans_mat.dot(self.mode_belief)

        #observational model P(o'|s')
        if not self.obs_tl:
            obs_likelihood = np.array([ np.exp(-self.evaluate_cost_to_go_(mode_idx, self.curr_state, self.curr_idx[mode_idx])) for mode_idx in range(self.n_modes) ])
        else:
            ti_cost_eval = np.array([ list(self.evaluate_cost_to_go_ti_(mode_idx, self.curr_state)) for mode_idx in range(self.n_modes) ])
            obs_likelihood = np.exp(-ti_cost_eval[:, 1])
            most_probable_phase_idx = ti_cost_eval[:, 0]


        #normalize
        obs_likelihood = obs_likelihood / np.sum(obs_likelihood)

        new_mode_belief = tmp_mode_belief * obs_likelihood
        #normalize
        new_mode_belief = new_mode_belief/np.sum(new_mode_belief)

        #update
        self.mode_belief = new_mode_belief

        #refresh the index if needed...
        if self.obs_tl:
            most_probable_mode = np.argmax(self.mode_belief)
            if most_probable_mode != self.curr_mode:
                self.curr_idx[most_probable_mode] = most_probable_phase_idx[most_probable_mode]
        return new_mode_belief

    def get_control_command(self):
        """
        get control command according to current observation and belief regarding the internal mode
        """
        #decision-making: maximum-posterior estimation instead of taking average...
        #the average decision might also work, but then that seems to be similar to other statistical model...
        most_probable_mode_idx = np.argmax(self.mode_belief)
        #derive controller with this local mode...
        #<hyin/Feb-7th-2016> note the current idx are a list of float phase variables...
        # ref_cmd = self.ref_trajs[most_probable_mode_idx][self.curr_idx]
        # ctrl_gain = self.feedback_gains[most_probable_mode_idx][self.curr_idx]
        ref_cmd, ctrl_gain = self.get_interp_fdfwd_and_fdbck_(most_probable_mode_idx, self.curr_idx[most_probable_mode_idx])
        return most_probable_mode_idx, ref_cmd, ctrl_gain

    def update(self, obs):
        """
        controller to wrap up the above perception/inference/decision subroutines
        """
        #<hyin/Feb-7th-2016> note the current idx are a list of float phase variables...
        #check if the current mode has finished or not...
        most_probable_mode_idx = np.argmax(self.mode_belief)

        if self.curr_idx[ most_probable_mode_idx ] > len(self.ref_trajs[0]):
            print 'Finish sending the trajectory command.'
            return None, None, None
        #first update belief according to the obs
        self.set_sensor_feedback(obs)
        self.update_mode_state()
        # print self.curr_idx
        self.curr_mode, ref_cmd, ctrl_gain = self.get_control_command()

        #update the index of command
        if not self.obs_tl:
            self.curr_idx += float(self.update_dt) / np.array(self.mode_phase_scales) * len(self.ref_trajs[0])
        else:
            #only consider current mode
            self.curr_idx[self.curr_mode] += float(self.update_dt) / np.array(self.mode_phase_scales[self.curr_mode]) * len(self.ref_trajs[0])


        return self.curr_mode, ref_cmd, ctrl_gain
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

    # traj_learner.pickle('test.pkl')
    return

import utils
import matplotlib.animation as animation

def EnCost_TrajAdaCtrl_Test(learner=None):
    n_samples = 20
    radius=0.5
    #a spatial circular path from [0, 0, 0] to [1, 1, 1]
    angulars = np.linspace(0, np.pi, 100)
    cir_path = np.array([
        radius * (1 - np.cos(angulars)) * np.cos(np.pi/4),
        radius * (1 - np.cos(angulars)) * np.sin(np.pi/4),
        radius * np.sin(angulars)
        ]).T
    #fit function approximators for the spatial path
    n_phases = 100
    z = np.linspace(0, 1.0, n_phases)
    fa_lst = [pyrbf_fa.PyRBF_FunctionApproximator(rbf_type='sigmoid', K=10, normalize=True) for dof in range(len(cir_path[0, :]))]
    dof_samples = []

    for dof in range(len(cir_path[0, :])):
        path_fa = fa_lst[dof]
        path_fa.set_linear_equ_constraints([z[0], z[-1]], [cir_path[0, dof], cir_path[-1, dof]])
        path_fa.fit(z, cir_path[:, dof], replace_theta=True)
        dof_samples.append(path_fa.gaussian_sampling(noise=0.001, n_samples=n_samples))

    #generate trajectories with different velocity...
    gen_trajs = []
    n_velgrp = 4
    n_grpsamples = n_samples/n_velgrp
    for i in range(n_velgrp):
        n_phases = 100 + i * 150
        test_z = np.linspace(0.0, 1.0, n_phases)
        sub_traj_lst = []
        for dof_idx, dof_theta_samples in enumerate(dof_samples):
            dof_theta_slices = dof_theta_samples[(i*n_grpsamples):((i+1)*n_grpsamples), :]
            #for these group of theta samples, evaluate corresponding to the desired time length
            tmp_dof_trajs = [fa_lst[dof_idx].evaluate(test_z, theta) for theta in dof_theta_slices]
            sub_traj_lst.append(tmp_dof_trajs)

        sub_traj_lst_recons = [np.array([sub_traj_lst[dof_idx][sample_idx] for dof_idx in range(len(fa_lst))]).T for sample_idx in range(n_grpsamples)]
        gen_trajs = gen_trajs + sub_traj_lst_recons
    #expand to also include the velocity as state
    aug_traj_data = utils.expand_traj_dim_with_derivative(gen_trajs)
    #interp again with fixed number of phases
    interp_data, phase_scale = utils.interp_data_fixed_num_phases(aug_traj_data, dt=0.01, num=100)

    #show the data...
    plt.ion()
    fig = plt.figure()
    ax_pos = fig.add_subplot(121, projection='3d')
    ax_vel = fig.add_subplot(122, projection='3d')
    ax_pos.hold(True)
    ax_vel.hold(True)
    ax_pos.set_aspect('equal')
    ax_vel.set_aspect('equal')
    # ax_pos.plot(xs=cir_path[:, 0], ys=cir_path[:, 1], zs=cir_path[:, 2], color='k', alpha=0.8)
    # ax_pos.plot(xs=aug_traj_data[5][:, 0], ys=aug_traj_data[5][:, 1], zs=aug_traj_data[5][:, 2], color='b')
    for traj, scale in zip(interp_data, phase_scale):
        ax_pos.plot(xs=traj[:, 0], ys=traj[:, 1], zs=traj[:, 2], linestyle='--', color='b', alpha=0.6)
        ax_vel.plot(xs=traj[:, 3], ys=traj[:, 4], zs=traj[:, 5], linestyle='-', color='b', alpha=0.6)

    plt.draw()

    #train a learner
    if learner is None:
        traj_learner = EnCost_TrajLearner(n_estimators=5, passive_type=None, verbose=True)
        traj_learner.train(data=interp_data, phase_scale=phase_scale)
    else:
        traj_learner = learner

    #plot reference trajectory for each mode
    transp = traj_learner.mode_weights * np.exp(-np.mean(np.array(traj_learner.cost_consts), axis=1)
        ) / np.amax(np.array(traj_learner.mode_weights) * 
        np.exp(-np.mean(np.array(traj_learner.cost_consts), axis=1)))

    # print traj_learner.mode_phase_scales
    vel_transp_exp = np.exp(-np.array(traj_learner.mode_phase_scales))
    vel_transp = vel_transp_exp / np.amax(vel_transp_exp)

    # for mode_idx, mean_traj in enumerate(traj_learner.mean_trajs):
    #     ax_pos.plot(xs=mean_traj[:, 0], ys=mean_traj[:, 1], zs=mean_traj[:, 2], color='k', linewidth=2.5, alpha=transp[mode_idx])
    #     ax_vel.plot(xs=mean_traj[:, 3], ys=mean_traj[:, 4], zs=mean_traj[:, 5], color='k', linewidth=2.5, alpha=vel_transp[mode_idx])
    # plt.draw()


    ctrl = EnCost_ImpController(learner=traj_learner, update_dt=0.01, obs_tl=True)

    ctrl.initialize_controller(mode=6, eigen_prob=0.9)
    realized_state = ctrl.curr_state
    ctrl.realized_state_storage = realized_state

    # print ctrl.mode_phase_scales
    step_cnt = 0

    realized_state_lst = []
    mode_belief_hist = []
    perturb_acc = -50*0
    dt = ctrl.update_dt
    while 1:
        curr_mode, ref_state, ctrl_gain = ctrl.update(realized_state)
        print 'current mode idx:', curr_mode
        if ref_state is None or ctrl_gain is None:
            break
        else:
            if step_cnt > 30 and step_cnt < 55:
                #apply some disturbance that can be exploit to adapt to another mode...
                #accelerate...
                realized_state = ref_state
                vel_dir = realized_state[3:] / np.linalg.norm(realized_state[3:])

                realized_state[3:] += vel_dir * dt * perturb_acc
                realized_state[:3] += .5 * vel_dir * perturb_acc * dt**2
                # print 'perturbation applied...'
            else:
                #assume the state command is perfectly realized
                realized_state = ref_state
            realized_state = ref_state
            step_cnt+=1
            realized_state_lst.append(realized_state)
            mode_belief_hist.append(ctrl.mode_belief)

    print 'use ', step_cnt, ' steps to execute the trajectory.'
    realized_traj = np.array(realized_state_lst)
    # draw the realized state...
    # ax_pos.plot(xs=realized_traj[:, 0], ys=realized_traj[:, 1], zs=realized_traj[:, 2], linewidth=3.0, linestyle='-', color='k', alpha=1.0)
    # ax_vel.plot(xs=realized_traj[:, 3], ys=realized_traj[:, 4], zs=realized_traj[:, 5], linewidth=3.0, linestyle='-', color='k', alpha=0.5)
    # plt.draw()

    raw_input('Press ENTER to start the animation...')
    pos_pnt, = ax_pos.plot(xs=[realized_traj[0, 0]], ys=[realized_traj[0, 1]], zs=[realized_traj[0, 2]], linewidth=3.0, linestyle='-', color='k', alpha=1.0)
    vel_pnt, = ax_vel.plot(xs=[realized_traj[0, 3]], ys=[realized_traj[0, 4]], zs=[realized_traj[0, 5]], linewidth=3.0, linestyle='-', color='k', alpha=1.0)
    for realized_state_idx in range(1, len(realized_state_lst), 5):
        #draw points...
        pos_pnt.set_xdata(realized_traj[:realized_state_idx, 0])
        pos_pnt.set_ydata(realized_traj[:realized_state_idx, 1])
        pos_pnt.set_3d_properties(realized_traj[:realized_state_idx, 2])
        vel_pnt.set_xdata(realized_traj[:realized_state_idx, 3])
        vel_pnt.set_ydata(realized_traj[:realized_state_idx, 4])
        vel_pnt.set_3d_properties(realized_traj[:realized_state_idx, 5])
        plt.pause(0.01)
        # ax_pos.plot(xs=realized_traj[:realized_state_idx, 0], ys=realized_traj[:realized_state_idx, 1], zs=realized_traj[:realized_state_idx, 2], markersize=5.0, linestyle='*', color='k')
        # ax_vel.plot(xs=realized_traj[:realized_state_idx, 3], ys=realized_traj[:realized_state_idx, 4], zs=realized_traj[:realized_state_idx, 5], markersize=5.0, linestyle='*', color='k')
        plt.draw()

    # # fig.canvas.draw()
    # ani = animation.FuncAnimation(fig, animate, len(realized_traj), interval=0.01, blit=False)
    plt.draw()

    return traj_learner

