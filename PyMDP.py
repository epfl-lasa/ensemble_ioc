"""
A simple module for solving model-based MDP problem
Discrete state/action space, known transition matrix...
A simplified python version similar to the one in Drake toolkit
(https://github.com/RobotLocomotion/drake)
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix

import PyMDP_Utils as utils

class DynamicalSystem:
    """
    A class to define a dynamical system: might be discrete time or continous time...
    """
    def __init__(self, is_ct=True):
        self.is_ct_ = is_ct
        # self.sysfunc_ = sysfunc         #a system function accepts state and control and
        #                                 #return derivative/next state, possibly with higher order stuff
        return
    def Dynamics(self, x, u, t=None, parms=None):
        print 'Calling base class dynamics...'
        return

    def IsGoalState(self, x):
        print 'Base class function to decide if this is the goal state'
        return False

    def RandomSelectFeasibleState(self, x):
        print 'Base class function to select a random feasible state'
        return False

class MarkovDecisionProcess:

    def __init__(self, S=None, A=None, T=None, C=None, gamma=1.0):
        self.S_ = S         # states:       if multiple dimension, then S[:, i] is state i
        self.A_ = A         # actions:      if multiple dimension, then A[:, i] is action i
        self.T_ = T         # transition:   T[k][i, j] = P(s_{n+1}=s[:, j] | s_{n}=s[:, i], a_{n}=[:, k])
        self.C_ = C         # cost:         cost C(s(:, i), a(:, k)) being at state i and taking action k
        self.gamma_ = gamma #discounted rate

        self.sub2ind_ = None    # for mapping multiple dimensional discrete state to flattened state space
        return

    def MakeSub2Ind(self, xbins):
        """
        function to generate a fast mapping from the vector of indices of dimensions to the flattened state space
        """
        nskip = np.concatenate([[1], np.cumprod([len(xbins[i]) for i in range(len(xbins)-1)])])
        self.sub2ind_ =  lambda subinds: np.sum(nskip*np.array(subinds))
        return self.sub2ind_

    def MakeXDigitize(self, xbins):
        """
        function to generate a mapping from a continous x to its digitized state according to the bins
        """
        #the first one will not be used
        effective_xbins = [xbin[1:] for xbin in xbins]
        self.xdigitize_ = lambda x: [np.digitize([dim], bin)[0] for dim, bin in zip(x, effective_xbins)]
        self.xdigitize_dim_ = lambda x, dim_idx: np.digitize([x[dim_idx]], effective_xbins[dim_idx])[0]
        return self.xdigitize_, self.xdigitize_dim_

    def DiscretizeSystem(self, sys, costfunc, xbins, ubins, options=dict()):
        #check system
        is_ct = sys.is_ct_

        #check dimension
        if not isinstance(xbins, list):
            #convert to multi-dimension case
            self.xbins_ = [xbins]
        else:
            self.xbins_ = xbins

        if not isinstance(ubins, list):
            self.ubins_ = [ubins]
        else:
            self.ubins_ = ubins

        self.state_dim_ = len(self.xbins_)
        self.ctrl_dim = len(self.ubins_)

        if 'dt' not in options:
            self.dt_ = 1.0
        else:
            self.dt_ = options['dt']

        if 'wrap_flag' not in options:
            self.wrap_flag_ = False * np.ones(self.state_dim_)
        else:
            self.wrap_flag_ = options['wrap_flag']

        wrap_idx = np.where(self.wrap_flag_==True)[0]

        xmin = np.array([bin[0] for bin in self.xbins_])
        xmax = np.array([bin[-1] for bin in self.xbins_])

        #construct grids
        #state
        Sgrid = np.meshgrid(*self.xbins_)

        #for each dim, need to reshape to a long 1-d array
        self.S_ = np.array([np.reshape(dim, (1, -1))[0] for dim in Sgrid])

        #action
        Agrid = np.meshgrid(*self.ubins_)
        self.A_ = np.array([np.reshape(dim, (1, -1))[0] for dim in Agrid])

        self.num_state_ = self.S_.shape[1]
        self.num_action_ = self.A_.shape[1]

        #prepare the transition matrix
        # self.T_ = csr_matrix([np.zeros([self.num_state_, self.num_state_]) for dim_ix in range(self.num_action_)])
        # self.T_ = [csr_matrix(np.zeros([self.num_state_, self.num_state_])) for dim_ix in range(self.num_action_)]
        self.T_ = [np.zeros([self.num_state_, self.num_state_]) for dim_ix in range(self.num_action_)]

        #prepare cost function
        self.C_ = np.zeros([self.num_state_, self.num_action_])

        #inline function to search index in reshaped state
        #offset for sub2ind
        sub2ind = self.MakeSub2Ind(self.xbins_)
        xdigitize, xdigitize_dim = self.MakeXDigitize(self.xbins_)

        print 'Constructing transition matrix...'
        #vectorize this to increase the efficiency if possible...
        for action_idx in range(self.num_action_):
            for state_idx in range(self.num_state_):
                if is_ct:
                    # the system must be an update equation
                    x_new = sys.Dynamics(self.S_[:, state_idx], self.A_[:, action_idx])
                    if np.isinf(costfunc(self.S_[:, state_idx], self.A_[:, action_idx], sys)) or np.isnan(costfunc(self.S_[:, state_idx], self.A_[:, action_idx], sys)):
                        print self.S_[:, state_idx], self.A_[:, action_idx]
                    else:
                        self.C_[state_idx, action_idx] = sys.dt_ * costfunc(self.S_[:, state_idx], self.A_[:, action_idx], sys)

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
                        x_new_mu_digitized_state = self.S_[:, sub2ind(x_new_mu_idx)]

                        coeff_lst = []
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
                                involved_states.append(involved_state_idx)
                                coeff_lst.append(np.exp(-np.linalg.norm(((self.S_[:, involved_state_idx] - x_new_mu_digitized_state)/x_new_sig))**2))

                        coeff_lst = coeff_lst / np.sum(coeff_lst)
                        #assign transition probability for each state
                        for coeff, involved_state_idx in zip(coeff_lst.tolist(), involved_states):
                            self.T_[action_idx][state_idx, involved_state_idx] += coeff
                    else:
                        #only updated state is available, need to map it to the grid
                        #add Baryinterpolation?
                        #wrap x_new if needed, this is useful for state variable like angular position
                        x_new[wrap_idx] = np.mod(x_new[wrap_idx] - xmin[wrap_idx],
                            xmax[wrap_idx] - xmin[wrap_idx]) + xmin[wrap_idx]

                        #barycentricinterpolation...
                        indices, coeffs = utils.BarycentricInterpolation(self.xbins_, np.array([x_new]))

                        for i in range(len(indices[0])):
                            self.T_[action_idx][state_idx, indices[0, i]] = coeffs[0, i]
                else:
                    #discrete state dynamical system...
                    #for discrete state dynamics, take the direct returned states and associated probability

                    x_new_lst = sys.Dynamics(self.S_[:, state_idx], self.A_[:, action_idx])
                    self.C_[state_idx, action_idx] = costfunc(self.S_[:, state_idx], self.A_[:, action_idx], sys)

                    for x_new in x_new_lst:
                        #get index of x_new
                        x_new_idx = xdigitize(x_new[0])
                        state_new_idx = sub2ind(x_new_idx)
                        self.T_[action_idx][state_idx, state_new_idx] = x_new[1]
        #check the T matrix
        # for action_idx in range(self.num_action_):
        #     if np.isinf(self.T_[action_idx]).any():
        #         print action_idx
        #         print np.isinf(self.T_[action_idx])
        return

    def ValueIteration(self, converged=.01, drawFunc=None, detail=False):
        J = np.zeros(self.num_state_)
        err = np.inf
        n_itrs = 0
        #prepare detailed result if necessary...
        if detail:
            res = dict()
            hist = []

        #vertical stack
        Tstack = np.vstack(self.T_)
        print 'Start Value Iteration...'
        ax = None

        while err > converged:
            Jold = np.array(J)
            #<hyin/Apr-14th-2015> note the reshape sequence of dot result
            J = np.amin(self.C_ + self.gamma_*np.reshape(Tstack.dot(J), (self.num_action_, self.num_state_)).transpose(), axis=1)
            # print 'iterating...'
            # if np.isinf(J).any() or np.isinf(Tstack).any():
            #     print np.isinf(J), np.isinf(Tstack)
            err = np.amax(np.abs(Jold-J))

            if detail:
                #record current itr, J, Q, err
                tmp_rec = dict()
                curr_Q = self.ActionValueFuncFromValueFunc(J)

                tmp_rec['value_func'] = J
                tmp_rec['action_value_func'] = curr_Q
                tmp_rec['error'] = err
                hist.append(tmp_rec)

            print 'Iteration:', n_itrs, 'Error:', err
            n_itrs+=1

            if drawFunc is not None:
                ax = drawFunc(self, J, ax)

        if detail:
            res['value_opt'] = hist[-1]['value_func']
            res['action_value_opt'] = hist[-1]['action_value_func']
            res['history'] = hist
            return res
        else:
            return J

    def ActionValueFuncFromValueFunc(self, J_opt):
        """
        Get Q action value function from optimal value function
        """
        def ActionValueValueIteration(T, C, J):
            Q = T.dot(C + self.gamma_*J)
            return Q

        Q_opt = [ActionValueValueIteration(self.T_[action_idx], self.C_[:, action_idx], J_opt) for action_idx in range(self.num_action_)]

        return Q_opt

    def ChooseOptimalActionFromQFunc(self, Q, state_idx):
        #helper function to choose optimal action from Q function
        #enumerate q values for all possible actions
        q = [Q[i][state_idx] for i in range(self.num_action_)]
        Q_min = min(q)
        count = q.count(Q_min)
        if count > 1:
            best = [i for i in range(self.num_action_) if q[i]==Q_min]
            action_idx = np.random.choice(best)
        else:
            action_idx = q.index(Q_min)

        return action_idx

    def QLearningSarsa(self, sys, epsilon=0.2, alpha=0.05, max_itrs=5000, drawFunc=None, detail=False):
        """
        learn Q from a dynamically learned policy
        alpha       - learning rate
        max_itrs    - number of steps to explore

        only support discrete state dynamical system
        """
        n_time_reach_goal = 0
        n_choose_optimal_action = 0
        n_steps_to_reach_goal = 0
        Q = [np.ones(self.num_state_)*0 for i in range(self.num_action_)]
        err = np.inf
        n_itrs = 0
        #prepare detailed result if necessary...
        if detail:
            res = dict()
            hist = []

        sub2ind = self.MakeSub2Ind(self.xbins_)
        xdigitize, xdigitize_dim = self.MakeXDigitize(self.xbins_)

        #generate a random initial state
        x = sys.RandomSelectFeasibleState()
        x_idx = xdigitize(x)
        state_idx = sub2ind(x_idx)

        #generate an action according to the epsilon greedy policy
        explore_dice = np.random.random_sample()
        if explore_dice < epsilon:
            #choose a random action
            action_idx = np.random.randint(low=0, high=self.num_action_)
        else:
            #greedy under current Q function
            action_idx = self.ChooseOptimalActionFromQFunc(Q, state_idx)
            n_choose_optimal_action += 1

        while n_itrs < max_itrs:
            #follow dynamics to get x_new, note the discrete dynamics returns
            #an array of new states and their associated probability
            x_new_lst = sys.Dynamics(self.S_[:, state_idx], self.A_[:, action_idx])
            c = self.C_[state_idx, action_idx]
            # print 'c:', c

            x_new_prob = [x_new[1] for x_new in x_new_lst]
            x_new_dice = np.argmax(np.random.multinomial(1, x_new_prob, size=1))
            x_new = x_new_lst[x_new_dice][0]
            x_new_idx = xdigitize(x_new)
            state_new_idx = sub2ind(x_new_idx)

            explore_dice_new = np.random.random_sample()
            if explore_dice_new < epsilon:
                #choose a random action
                action_new_idx = np.random.randint(low=0, high=self.num_action_)
            else:
                #greedy under current Q function
                action_new_idx = self.ChooseOptimalActionFromQFunc(Q, state_new_idx)
                # print 'Choose current optimal action!', action_new_idx
                n_choose_optimal_action += 1

            #update curr Q value for current state index and action index
            if Q[action_idx][state_idx] == np.inf:
                Q[action_idx][state_idx] = c
            else:
                Q[action_idx][state_idx] += alpha*(c + self.gamma_*Q[action_new_idx][state_new_idx] - Q[action_idx][state_idx])

            #check if new state is a terminal one...
            if sys.IsGoalState(x_new):
                # print 'Used ', n_steps_to_reach_goal, ' to reach the goal.'
                # raw_input()
                n_steps_to_reach_goal=0
                n_time_reach_goal += 1
                # raw_input()
                #zero costs for x_new as it is the goal state
                for action_idx in range(self.num_action_):
                    Q[action_idx][state_new_idx] = 0.0
                #a new random state
                x = sys.RandomSelectFeasibleState()
                x_idx = xdigitize(x)
                state_idx = sub2ind(x_idx)

                explore_dice = np.random.random_sample()
                if explore_dice < epsilon:
                    #choose a random action
                    action_idx = np.random.randint(low=0, high=self.num_action_)
                else:
                    #greedy under current Q function
                    action_idx = self.ChooseOptimalActionFromQFunc(Q, state_idx)
                    # print 'Choose current optimal action!', action_idx
                    n_choose_optimal_action += 1

            else:
                state_idx = state_new_idx
                action_idx = action_new_idx

            if detail:
                tmp_rec = dict()
                tmp_rec['action_value_func'] = np.array(Q, copy=True)
                hist.append(tmp_rec)

            print 'Iteration:', n_itrs
            n_itrs+=1
            n_steps_to_reach_goal+=1

        print 'Times of reaching the goal:', n_time_reach_goal
        print 'Times of choosing optimal action', n_choose_optimal_action

        if detail:
            res['action_value_opt'] = hist[-1]['action_value_func']
            res['history'] = hist
            return res
        else:
            return Q
        return
    def QLearningEpsilonGreedy(self, sys, epsilon=0.2, alpha=0.05, max_itrs=5000, drawFunc=None, detail=False):
        """
        learn Q given an initial guess and epsilon greedy policy
        epsilon     - probability to deviate from the greedy policy to explore
        alpha       - learning rate
        max_itrs    - number of steps to explore

        only support discrete state dynamical system
        """
        Q = [np.ones(self.num_state_)*0 for i in range(self.num_action_)]
        err = np.inf
        n_itrs = 0
        #prepare detailed result if necessary...
        if detail:
            res = dict()
            hist = []

        sub2ind = self.MakeSub2Ind(self.xbins_)
        xdigitize, xdigitize_dim = self.MakeXDigitize(self.xbins_)

        #generate a random initial state
        x = sys.RandomSelectFeasibleState()
        x_idx = xdigitize(x)
        state_idx = sub2ind(x_idx)

        while n_itrs < max_itrs:
            #generate an action according to the epsilon greedy policy
            #throw a dice
            explore_dice = np.random.random_sample()
            if explore_dice < epsilon:
                #choose a random action
                action_idx = np.random.randint(low=0, high=self.num_action_)
            else:
                #greedy under current Q function
                action_idx = self.ChooseOptimalActionFromQFunc(Q, state_idx)

            #follow dynamics to get x_new, note the discrete dynamics returns
            #an array of new states and their associated probability
            x_new_lst = sys.Dynamics(self.S_[:, state_idx], self.A_[:, action_idx])
            c = self.C_[state_idx, action_idx]

            x_new_prob = [x_new[1] for x_new in x_new_lst]
            print 'x_new_prob:', x_new_prob
            x_new_dice = np.argmax(np.random.multinomial(1, x_new_prob, size=1))
            x_new = x_new_lst[x_new_dice][0]
            print 'x_new:', x_new
            x_new_idx = xdigitize(x_new)
            state_new_idx = sub2ind(x_new_idx)

            #update curr Q value for current state index and action index
            if Q[action_idx][state_idx] == np.inf:
                Q[action_idx][state_idx] = c
            else:
                Q[action_idx][state_idx] += alpha*(c + self.gamma_*min([Q[i][state_new_idx]
                    for i in range(self.num_action_)]) - Q[action_idx][state_idx])

            #check if new state is a terminal one...
            if sys.IsGoalState(x_new):
                #zero costs for x_new as it is the goal state
                for action_idx in range(self.num_action_):
                    Q[action_idx][state_new_idx] = 0.0
                #a new random state
                x = sys.RandomSelectFeasibleState()
                x_idx = xdigitize(x)
                state_idx = sub2ind(x_idx)
            else:
                state_idx = state_new_idx

            if detail:
                tmp_rec = dict()
                tmp_rec['action_value_func'] = np.array(Q, copy=True)
                hist.append(tmp_rec)

            print 'Iteration:', n_itrs
            n_itrs+=1

        if detail:
            res['action_value_opt'] = hist[-1]['action_value_func']
            res['history'] = hist
            return res
        else:
            return Q
