"""
Test sample of Inverted Pendulum
"""

from PyMDP import *
import PyMDP_Utils as utils
import ensemble_ioc as eioc

#pendulum test
class PendulumDynSys(DynamicalSystem):

    #parameters
    m_ = 1
    l_ = .5
    b_ = .1
    lc_ = .5
    g_ = 9.81
    dt_ = 0.01

    sigma_ = 0.1

    def __init__(self, dt=0.01):
        DynamicalSystem.__init__(self, is_ct=True)
        self.dt_ = dt
        self.I_ = self.m_ * self.l_**2
        return

    def PassiveDynamics(self, x, t=None, parms=None):
        '''
        passive dynamics where no control is given. This is feasible for pendulum DynamicalSystem
        as it is control affine
        '''
        x_new=self.Dynamics(x=x, u=np.zeros(1), t=t, parms=parms)
        return x_new

    def Dynamics(self, x, u, t=None, parms=None):
        """
        A second-order dynamics
        """
        qdd = (u[0] - self.m_*self.g_*self.lc_*np.sin(x[0]) - self.b_*x[1])/self.I_

        # dont need term +0.5*(qdd**2)*self.dt_?
        x_new =  x + self.dt_ * np.array([x[1], qdd])
        # do we need to compensate a small part of grid for the positive direction?
        # <hyin/Apr-16th-2015> the argument for this fix might be, our discretization is defined as
        #   the i-th state x_{i-1} <= x < x_i and all the i-th for dyanmics evluation is actually x_{-1}
        #   This makes state decreasing easy and increasing hard as the control input must climb up
        #   the full interval. However, I would like to ascribe this as a temporary explanation because
        #   I didnt see any sample code with really special care about the discretization
        #   the ugly fix makes the lower half looks better but I wont say it gives the correct profile
        #   I dont know either if Baryinterpolation is the killer as we also have some uncertainty here so its not a deterministic dynamics
        #   need careful examination of Russ' code one more time...
        #   also an interesting thing is that quadratic cost has less asymetrical effect, even with zero control cost
        #   IMPORTANT obervation: lots of parameters give cost value that does not make sense. They tend to score
        #   state [0, 0] or [2*np.pi, 0] as an easy one, but actually non-zero velocity should be preferred in this case
        #   only one type of torque range gives this expected effects and increasing the limits of torque surprisingly
        #   removes this correct feature? How can be like that?
        # if x[1] > 0:
        #     x_new += np.array([2*np.pi/51 * 1, 5.0/51 * 1])
        # <hyin/Apr-18th-2015> the above one give result that looks same to Russ' example 'qualitatively'.
        # but the red region eats some part of the blue one. And moreover, the mintime cost is almost same as the quadratic one...
        # I think Baryinterpolation might be a necessary to prevent the above compensation term...
        # return [x_new, [self.sigma_*0.1, self.sigma_]]
        # <hyin/Apr-19th-2015> by adding BarycentricInterpolation, there seems to be 99% similarity between the results
        # of ours and Russ' for lqr cost. However, there still seems to be problems with the mintime example. Concretely,
        # the [0, 0] and [np.pi*2, 0] is not that dark. Is there any difference between our dynamics or the implementation of the interpolation?
        # let's first use the lqr example... The compensation term is deprecated now but our naive approach for stochastic dynamics
        # seems to lead to zero-order approximation, which is probably the root cause of asymmetry.
        return x_new

def PendulumDynTest():

    pendulum = PendulumDynSys()

    x = [1.0/4*np.pi, 0]
    dt = 0.01
    time_step = 1000

    def pendulum_draw_func(x, ax=None, pend=None):
        plt.ion()

        len_coord = np.linspace(0.0, 0.5, 100)
        pnts = [np.cos(x[0]-np.pi/2)*len_coord, np.sin(x[0]-np.pi/2)*len_coord]
        if ax is None or pend is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(0, 0, 'go', linewidth=4.0)
            pend, = ax.plot(pnts[0], pnts[1], linewidth=2.0)
            plt.xlim([-1.5, 1.5])
            plt.ylim([-2.5, 1.5])
            plt.show()
        else:
            pend.set_xdata(pnts[0])
            pend.set_ydata(pnts[1])

        plt.draw()
        # time.sleep(1)
        raw_input()
        return ax, pend

    ax = None
    pend = None
    for i in range(time_step):
        x_old = np.array(x)
        x = pendulum.Dynamics(x, 0)
        print x

        ax, pend = pendulum_draw_func(x_old, ax, pend)

    return

def PendulumTransDynDraw(pendulumMDP, a_idx):
    #draw the transition matrix for given action
    trans_mat = pendulumMDP.T_[a_idx]
    links = []
    for row_idx in range(trans_mat.shape[0]):
        for col_idx in range(trans_mat.shape[1]):
            #for each element, find the old state and new state
            if trans_mat[row_idx, col_idx] > 0.5:
                old_state = pendulumMDP.S_[:, row_idx]
                new_state = pendulumMDP.S_[:, col_idx]
                links.append([old_state, new_state, trans_mat[row_idx, col_idx]])
    #prepare a figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hold(True)
    for link in links:
        ax.plot([link[0][0], link[1][0]], [link[0][1], link[1][1]], '-bo')

    plt.draw()
    return

import time

def PendulumTransitionTest():
    def pendulum_lqr_cost(x, u, sys=None):
        xd = np.array([np.pi, 0])

        c = np.linalg.norm((x-xd)*np.array([1, 1]))**2 + np.linalg.norm(u)**2
        return c

    pendulum = PendulumDynSys(dt=0.01)
    pendulumMDP = MarkovDecisionProcess(gamma=0.95)
    xbins = [np.linspace(0, 2*np.pi, 51), np.linspace(-2.5, 2.5, 51)]
    ubins = [np.linspace(-5.0, 5.0, 21)]
    action_idx = 10

    options = dict()
    options['dt'] = pendulum.dt_
    options['wrap_flag'] = np.array([True, False])

    pendulumMDP.DiscretizeSystem(pendulum, pendulum_lqr_cost, xbins, ubins, options)

    wrap_idx = np.where(pendulumMDP.wrap_flag_==True)[0]
    sub2ind = pendulumMDP.sub2ind_
    xdigitize = pendulumMDP.xdigitize_
    xdigitize_dim = pendulumMDP.xdigitize_dim_
    xmin = np.array([bin[0] for bin in pendulumMDP.xbins_])
    xmax = np.array([bin[-1] for bin in pendulumMDP.xbins_])

    #check T_
    for state_idx in range(pendulumMDP.num_state_):
        #if np.abs((pendulumMDP.S_[:, state_idx][0]-5.15221195))<1e-1 and np.abs((pendulumMDP.S_[:, state_idx][1]-2.5))<1e-1:
        if np.abs((pendulumMDP.S_[:, state_idx][0]-np.pi/4))<1e-1 and np.abs((pendulumMDP.S_[:, state_idx][1]-0))<1e-1:
            x_old_idx = xdigitize(pendulumMDP.S_[:, state_idx])
            print pendulumMDP.S_[:, state_idx], sub2ind(x_old_idx), pendulumMDP.S_[:, sub2ind(x_old_idx)]
            x_new = pendulum.Dynamics(pendulumMDP.S_[:, state_idx], pendulumMDP.A_[:, action_idx])
            if isinstance(x_new, list):
                #contains both expected state and diagonal Gaussian noise...
                x_new_mu = x_new[0]
                x_new_sig = x_new[1]

                if len(x_new_mu) != len(x_new_sig):
                    print 'Inconsistent length of state and noise vector...'
                    return
                #wrap x_new if needed, this is useful for state variable like angular position
                x_new_mu[wrap_idx] = np.mod(x_new_mu[wrap_idx] - xmin[wrap_idx], 
                    xmax[wrap_idx] - xmin[wrap_idx]) + xmin[wrap_idx]
                x_new_mu_idx = xdigitize(x_new_mu)
                x_new_mu_digitized_state = pendulumMDP.S_[:, sub2ind(x_new_mu_idx)]

                involved_inds = []
                involved_states = []
                for dim_idx in range(len(x_new_mu)):
                    tmp_x_new_mu_idx = [idx for idx in x_new_mu_idx]
                    #for each dim, try to crawl the grid
                    #find lower bound, use the interval [-2*sigma, 2*sigma]
                    #how to wrap here? or just truncate the shape of gaussian?...
                    x_new_mu_tmp_min = np.array(x_new_mu)
                    x_new_mu_tmp_max = np.array(x_new_mu)
                    x_new_mu_tmp_min[dim_idx] += -2*x_new_sig[dim_idx]
                    x_new_mu_tmp_max[dim_idx] +=  2*x_new_sig[dim_idx]
                    min_idx = xdigitize_dim(x_new_mu_tmp_min, dim_idx)
                    max_idx = xdigitize_dim(x_new_mu_tmp_max, dim_idx)
                    
                    for step_idx in range(min_idx, max_idx+1):
                        tmp_x_new_mu_idx[dim_idx] = step_idx
                        #get the index of involved state
                        involved_state_idx = sub2ind(tmp_x_new_mu_idx)
                        involved_inds.append(involved_state_idx)
                        involved_states.append(pendulumMDP.S_[:, involved_state_idx])

                print x_new, involved_inds, involved_states
                for state_new_idx in involved_inds:
                    print pendulumMDP.T_[action_idx][state_idx, state_new_idx]
            else:
                x_new_idx = xdigitize(x_new)
                state_new_idx = sub2ind(x_new_idx)
                print x_new, x_new_idx, state_new_idx, pendulumMDP.S_[:, state_new_idx]
                print pendulumMDP.T_[action_idx][state_idx, state_new_idx]

            raw_input()

    PendulumTransDynDraw(pendulumMDP, action_idx)

    #see how's the transition matrix working...
    #passive dynamics, T[4]?
    x = [1.0/4*np.pi, 0]

    dt = 0.05
    time_step = 100

    def pendulum_draw_func(x, ax=None, pend=None):
        plt.ion()

        len_coord = np.linspace(0.0, 0.5, 100)
        pnts = [np.cos(x[0]-np.pi/2)*len_coord, np.sin(x[0]-np.pi/2)*len_coord]
        if ax is None or pend is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(0, 0, 'go', linewidth=4.0)
            pend, = ax.plot(pnts[0], pnts[1], linewidth=2.0)
            plt.xlim([-1.5, 1.5])
            plt.ylim([-2.5, 1.5])
            plt.show()
        else:
            pend.set_xdata(pnts[0])
            pend.set_ydata(pnts[1])

        plt.draw()
        # time.sleep(1)
        raw_input()
        return ax, pend

    ax = None
    pend = None

    x_old = np.array(x)
    x_old_from_dyn = np.array(x_old)
    x_old_idx = xdigitize(x_old)
    state_old_idx = sub2ind(x_old_idx)
    print state_old_idx, pendulumMDP.S_[:, state_old_idx]

    for i in range(time_step):
        
        x_new_from_dyn = pendulum.Dynamics(x_old_from_dyn, [0])
        if isinstance(x_new_from_dyn, list):
            x_new_from_dyn = x_new_from_dyn[0]
        else:
            pass

        #extract idx
        # x_old_idx = [np.digitize([dim], bin)[0]-1 for dim, bin in zip(x_old, pendulumMDP.xbins_)]
        # print pendulumMDP.T_[4][state_old_idx, :][pendulumMDP.T_[4][state_old_idx, :]>0]
        state_new_idx = np.argmax(pendulumMDP.T_[action_idx][state_old_idx, :])
        # print pendulumMDP.T_[action_idx][state_old_idx, :][pendulumMDP.T_[action_idx][state_old_idx, :]>0.5]
        #get x_new
        x = pendulumMDP.S_[:, state_new_idx]
        print x, x_new_from_dyn
        ax, pend = pendulum_draw_func(pendulumMDP.S_[:, state_old_idx], ax, pend)
        state_old_idx = state_new_idx
        x_old_from_dyn = x_new_from_dyn

    return

def PendulumTrucateState(x, xmin, xmax, wrap_idx):
    x_new = np.array(x)
    # it seems this complicates the problem...
    # we need to be very careful when caculating mean of states...
    # x_new[wrap_idx] = np.mod(x_new[wrap_idx] - xmin[wrap_idx], 
    #                 xmax[wrap_idx] - xmin[wrap_idx]) + xmin[wrap_idx]
    for dim_idx in range(len(x)):
        if x_new[dim_idx] > xmax[dim_idx]:
            x_new[dim_idx] = xmax[dim_idx]
        elif x_new[dim_idx] < xmin[dim_idx]:
            x_new[dim_idx] = xmin[dim_idx]
    return x_new

def PendulumMDPTrajOpt(sys, mdp, Q_opt, x0, T=200):
    #for given x0...
    traj = [x0]
    sub2ind = mdp.sub2ind_
    xdigitize = mdp.xdigitize_
    xmin = np.array([bin[0] for bin in mdp.xbins_])
    xmax = np.array([bin[-1] for bin in mdp.xbins_])
    wrap_idx = np.where(mdp.wrap_flag_==True)[0]

    for i in range(T):
        #get control
        state_idx = sub2ind(xdigitize(traj[-1]))
        action_idx = np.argmin([Q_a[state_idx] for Q_a in Q_opt])
        #get new state
        x_new = sys.Dynamics(traj[-1], mdp.A_[:, action_idx])
        #truncate x_new...
        # x_new = PendulumTrucateState(x_new, xmin, xmax, wrap_idx)
        #append this new state
        traj.append(x_new)
        # check if reached limit, if yes, stop generate trajectory
        # if x_new[0] < xmin[0] or x_new[0] > xmax[0] or x_new[1] < xmin[1] or x_new[1] > xmax[1]:
        #     break

    return traj

def pendulum_traj_draw(traj, ax=None):
    plt.ion()
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    traj_array = np.array(traj)
    ax.hold(True)
    line = ax.plot(traj_array[:, 0], traj_array[:, 1], '-k')
    #add larger marker for the initial point
    ax.plot(traj_array[0, 0], traj_array[0, 1], '*k', markersize=10.0)
    #add arrow to curve
    utils.add_arrow_to_line2D(ax, line)
    return ax

def PendulumMDPTest():
    
    def pendulum_lqr_cost(x, u, sys=None):
        xd = np.array([np.pi, 0])

        c = np.linalg.norm((x-xd)*np.array([1, 1]))**2 + np.linalg.norm(u)**2

        return c

    def pendulum_mintime_cost(x, u, sys=None):
        xd = np.array([np.pi, 0])

        thres = 0.05
        if np.linalg.norm((x-xd)*np.array([1, 1]))**2 < thres:
            c = 0 
        else:
            c = 1 
        return c

    def pendulum_value_func_draw(mdp, J, ax=None):

        plt.ion()
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.pcolormesh(mdp.xbins_[0], mdp.xbins_[1], 
            np.reshape(J, (len(mdp.xbins_[1]), len(mdp.xbins_[0]))),
            shading='none')
        ax.set_xlabel('x (rad)', fontsize=16)
        ax.set_ylabel('x_dot (rad/s)', fontsize=16)
        xmin = np.array([bin[0] for bin in mdp.xbins_])
        xmax = np.array([bin[-1] for bin in mdp.xbins_])
        ax.set_xlim([xmin[0], xmax[0]])
        ax.set_ylim([xmin[1], xmax[1]])
        plt.draw()

        return ax

    pendulum = PendulumDynSys(dt=0.01)
    pendulumMDP = MarkovDecisionProcess(gamma=0.999)

    cost_func = pendulum_lqr_cost
    ulim = 5.0

    # cost_func = pendulum_mintime_cost
    # ulim = 1.0

    xbins = [np.linspace(0, 2*np.pi, 51), np.linspace(-10, 10, 51)]
    ubins = [np.linspace(-ulim, ulim, 9)]

    options = dict()
    options['dt'] = pendulum.dt_
    options['wrap_flag'] = np.array([True, False])

    pendulumMDP.DiscretizeSystem(pendulum, cost_func, xbins, ubins, options)

    #test value iteration
    opt_res = pendulumMDP.ValueIteration(converged=0.001, drawFunc=None, detail=True)

    J_opt = opt_res['value_opt']
    Q_opt = opt_res['action_value_opt']

    ax = pendulum_value_func_draw(pendulumMDP, J_opt, ax=None)
    
    #generate some trajectories from Q_opt
    N = 500
    x0_lst = np.random.random((N, 2))
    x0_lst[:, 0] = x0_lst[:, 0] * np.pi*2
    x0_lst[:, 1] = x0_lst[:, 1]*10*2 - 10
    # uniform sampling of data
    # x0_lst = pendulumMDP.S_.T
    traj_opt = [PendulumMDPTrajOpt(pendulum, pendulumMDP, Q_opt, x0, T=200) for x0 in x0_lst]
    # raw_input('Press ENTER to add generated trajectories...')
    #update trajectory on cost figure
    # for traj in traj_opt[0:30]:
    #     pendulum_traj_draw(traj, ax)

    opt_res['traj_opt'] = traj_opt
    return opt_res

from sklearn.ensemble import RandomTreesEmbedding
from collections import defaultdict
from scipy.misc import logsumexp

def PendulumMDPValueLearning(traj_lst, sys, n_est=50, rs=0, mdp=None):
    data = []
    ax=None
    xmin = np.array([0, -10])
    xmax = np.array([2*np.pi, 10])
    for traj in traj_lst[0:]:
        ax = pendulum_traj_draw(traj, ax)
        noise_array = np.random.normal(scale=0.02,size=(len(traj), 2))
        for idx in range(0, len(traj)-100):
            #check if the data is within the interested range
            if traj[idx][0] < xmin[0] or traj[idx][0] > xmax[0] or traj[idx][1] < xmin[1] or traj[idx][1] > xmax[1] \
                or traj[idx+1][0] < xmin[0] or traj[idx+1][0] > xmax[0] or traj[idx+1][1] < xmin[1] or traj[idx+1][1] > xmax[1]:
                continue
            else:
                data.append(np.concatenate([traj[idx]+noise_array[idx], traj[idx+1]+noise_array[idx+1]]))     #demonstrated x_n, x_{n+1}
    data=np.array(data)
    #train with EnsembleIOC
    mdl=eioc.EnsembleIOC(n_estimators=n_est, max_depth=3, em_itrs=5, min_samples_split=10, min_samples_leaf=5, regularization=1e-5, 
        passive_dyn_func=sys.PassiveDynamics, passive_dyn_ctrl=np.array([[0, 0], [0, 1]]), passive_dyn_noise=1e-3, verbose=True)
    mdl.fit(X=data)

    #construct value function
    def res_val(x, mdl):
        vals = mdl.value_eval_samples(np.array([x]),average=False)
        return vals[0]

    return res_val, mdl

def PendulumMDPValueLearningTest(opt_res):
    pendulum = PendulumDynSys(dt=0.01)
    traj_lst = opt_res['traj_opt']
    value_func, rf_mdl = PendulumMDPValueLearning(traj_lst, sys=pendulum)

    #test to show values...
    x = np.linspace(0, 2*np.pi, 51)
    x_dot = np.linspace(-10, 10, 51)
    Sgrid = np.meshgrid(x, x_dot)
    states = np.array([np.reshape(dim, (1, -1))[0] for dim in Sgrid])
    J = np.zeros(states.shape[1])
    for state_idx in range(states.shape[1]):
        J[state_idx] = value_func(states[:, state_idx], rf_mdl)
    print J
    def pendulum_value_func_draw(x, x_dot, J, ax=None):

        plt.ion()
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        pcol = ax.pcolormesh(x, x_dot, 
            np.reshape(J, (len(x_dot), len(x))),
            shading='none')
        pcol.set_edgecolor('face')
        ax.set_xlabel('x (rad)', fontsize=16)
        ax.set_ylabel('x_dot (rad/s)', fontsize=16)
        xmin = np.array([bin[0] for bin in [x, x_dot]])
        xmax = np.array([bin[-1] for bin in [x, x_dot]])
        ax.set_xlim([xmin[0], xmax[0]])
        ax.set_ylim([xmin[1], xmax[1]])
        plt.draw()

        return ax

    pendulum_value_func_draw(x, x_dot, J)
    return

def PendulumMDPValueLearningError(opt_res):

    M = [3, 5, 8, 15, 20, 30, 50, 100, 150, 200]
    pendulum = PendulumDynSys(dt=0.01)
    traj_lst = opt_res['traj_opt']
    J_opt = opt_res['value_opt']

    J_opt_normed = (J_opt - np.mean(J_opt)) 

    err_lst = []
    run_num_itrs = 5
    for m in M:
        err = []
        for i in range(run_num_itrs):
            value_func, rf_mdl = PendulumMDPValueLearning(traj_lst, sys=pendulum, n_est=m, rs=i)

            #test to show values...
            x = np.linspace(0, 2*np.pi, 51)
            x_dot = np.linspace(-10, 10, 51)
            Sgrid = np.meshgrid(x, x_dot)
            states = np.array([np.reshape(dim, (1, -1))[0] for dim in Sgrid])
            J = np.zeros(states.shape[1])
            for state_idx in range(states.shape[1]):
                J[state_idx] = value_func(states[:, state_idx], rf_mdl)
            #norm this J
            J_normed = (J - np.mean(J))
            err.append(np.linalg.norm(J_normed - J_opt_normed, ord=1)/len(J_opt))
        err_lst.append(err)

    #plot
    err_lst = np.array(err_lst)
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(M, np.mean(err_lst, axis=1), yerr=np.std(err_lst, axis=1), fmt='-o', linewidth=4.0, markersize=20.0)
    ax.set_xlabel('Model size - M', fontsize=20)
    ax.set_ylabel('Cost err', fontsize=20)
    ax.set_title('Cost Error versus Model Size', fontsize=20)

    plt.draw()
    return

import timeit

def PendulumMDPValueLearningTimeCost(opt_res, time_rec=None):
    # M = [5, 10, 20]
    M = [5, 10, 20, 30, 50, 80, 100, 150, 200]
    pendulum = PendulumDynSys(dt=0.01)
    traj_lst = opt_res['traj_opt']
    run_num_itrs = 5
    if time_rec is None:
        time_elapsed = []
        for m in M:
            def test_func():
                return PendulumMDPValueLearning(traj_lst, sys=pendulum, n_est=m)
            #do value iteration and record the time consumption
            time_cost_lst = []
            for i in range(run_num_itrs):
                time_cost = timeit.timeit(test_func, number=1)
                time_cost_lst.append(time_cost)
            time_elapsed.append(time_cost_lst)
    else:
        time_elapsed = time_rec

    #plot
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    time_elapsed_array = np.array(time_elapsed)
    print np.std(time_elapsed_array, axis=1)
    ax.errorbar(M, np.mean(time_elapsed, axis=1), yerr=np.std(time_elapsed_array, axis=1), fmt='-o', linewidth=4.0, markersize=20.0)
    ax.set_xlabel('Model size - M', fontsize=20)
    ax.set_ylabel('Time cost (sec)', fontsize=20)
    ax.set_title('Time Complexity versus Model Size', fontsize=20)

    plt.draw()
    return time_elapsed