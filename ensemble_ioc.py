"""
A module that implements the ensemble of inverse optimal control models
"""
import cPickle as cp
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn import decomposition
from sklearn.ensemble import RandomTreesEmbedding
from sklearn import mixture
from scipy.misc import logsumexp

EPS = np.finfo(float).eps

class EnsembleIOC(BaseEstimator, RegressorMixin):

    def __init__(self,  n_estimators=20, 
                        max_depth=5, min_samples_split=10, min_samples_leaf=10,
                        random_state=0,
                        em_itrs=5,
                        regularization=0.05,
                        passive_dyn_func=None,
                        passive_dyn_ctrl=None,
                        passive_dyn_noise=None,
                        verbose=False):
        '''
        n_estimators        - number of ensembled models
        ...                 - a batch of parameters used for RandomTreesEmbedding, see relevant documents
        em_itrs             - maximum number of EM iterations to take
        regularization      - small positive scalar to prevent singularity of matrix inversion
        passive_dyn_func    - function to evaluate passive dynamics; None for MaxEnt model
        passive_dyn_ctrl    - function to return the control matrix which might depend on the state...
        passive_dyn_noise   - covariance of a Gaussian noise; only applicable when passive_dyn is Gaussian; None for MaxEnt model
                                note this implies a dynamical system with constant input gain. It is extendable to have state dependent
                                input gain then we need covariance for each data point
        verbose             - output training information
        '''
        BaseEstimator.__init__(self)

        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf
        self.random_state=random_state
        self.em_itrs=em_itrs
        self.reg=regularization
        self.passive_dyn_func=passive_dyn_func
        self.passive_dyn_ctrl=passive_dyn_ctrl
        self.passive_dyn_noise=passive_dyn_noise
        self.verbose=verbose
        return

    def fit(self, X, y=None):
        '''
        y could be the array of starting state of the demonstrated trajectories/policies
        if it is None, it implicitly implies a MaxEnt model. Other wise, it serves as the feature mapping
        of the starting state. This data might also be potentially used for learning the passive dynamics
        for a pure model-free learning with some regressors and regularization.
        '''
        #check parameters...
        assert(type(self.n_estimators)==int)
        assert(self.n_estimators > 0)
        assert(type(self.max_depth)==int)
        assert(self.max_depth > 0)
        assert(type(self.min_samples_split)==int)
        assert(self.min_samples_split > 0)
        assert(type(self.min_samples_leaf)==int)
        assert(self.min_samples_leaf > 0)
        assert(type(self.em_itrs)==int)

        #an initial partitioning of data with random forest embedding
        self.random_embedding_mdl_ = RandomTreesEmbedding(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
            )

        #we probably do not need the data type to differentiate it is a demonstration
        #of trajectory or commanded state, do we?
        if self.passive_dyn_func is not None and self.passive_dyn_ctrl is not None and self.passive_dyn_noise is not None:
            self.random_embedding_mdl_.fit(X[:, X.shape[1]/2:])
            indices = self.random_embedding_mdl_.apply(X[:, X.shape[1]/2:])
            # X_tmp = np.array(X)
            # X_tmp[:, X.shape[1]/2:] = X_tmp[:, X.shape[1]/2:] - X_tmp[:, :X.shape[1]/2]
            # self.random_embedding_mdl_.fit(X_tmp)

            # indices = self.random_embedding_mdl_.apply(X_tmp)
        else:
            self.random_embedding_mdl_.fit(X)
            #figure out indices
            indices = self.random_embedding_mdl_.apply(X)

        partitioned_data = defaultdict(list)

        leaf_idx = defaultdict(set)
        weight_idx = defaultdict(float)
        #group data belongs to the same partition and have the weights...
        #is weight really necessary for EM steps? Hmm, seems to be for the initialization
        #d_idx: data index; p_idx: partition index (comprised of estimator index and leaf index)
        for d_idx, d, p_idx in zip(range(len(X)), X, indices):
            for e_idx, l_idx in enumerate(p_idx):
                partitioned_data[e_idx, l_idx].append(d)
                leaf_idx[e_idx] |= {l_idx}

            for e_idx, l_idx in enumerate(p_idx):
                weight_idx[e_idx, l_idx] = float(len(partitioned_data[e_idx, l_idx])) / len(X)
                # weight_idx[e_idx, l_idx] = 1. / len(p_idx)

        #for each grouped data, solve an easy IOC problem by assuming quadratic cost-to-go function
        #note that, if the passive dynamics need to be learned, extra steps is needed to train a regressor with weighted data
        #otherwise, just a simply gaussian for each conditional probability distribution model
        self.estimators_ = []
        #another copy to store the parameters all together, for EM/evaluation on all of the models
        self.estimators_full_ = defaultdict(list)
        #<hyin/Feb-6th-2016> an estimator and leaf indexed structure to record the passive likelihood of data...
        passive_likelihood_dict = defaultdict(list)
        for e_idx in range(self.n_estimators):
            #for each estimator
            estimator_parms = defaultdict(list)
            for l_idx in leaf_idx[e_idx]:
                if self.verbose:
                    print 'Processing {0}-th estimator and {1}-th leaf...'.format(e_idx, l_idx)
                #and for each data partition
                data_partition=np.array(partitioned_data[e_idx, l_idx])
                if self.passive_dyn_func is not None and self.passive_dyn_ctrl is not None and self.passive_dyn_noise is not None:
                    X_new         = data_partition[:, data_partition.shape[1]/2:]
                    X_old         = data_partition[:, 0:data_partition.shape[1]/2]
                    X_new_passive = np.array([self.passive_dyn_func(X_old[sample_idx]) for sample_idx in range(data_partition.shape[0])])
                    passive_likelihood = _passive_dyn_likelihood(X_new, X_new_passive, self.passive_dyn_noise, self.passive_dyn_ctrl, self.reg)

                    weights = passive_likelihood / np.sum(passive_likelihood)
                    weighted_mean = np.sum((weights*X_new.T).T, axis=0)

                    estimator_parms['means'].append(weighted_mean)
                    estimator_parms['covars'].append(_frequency_weighted_covariance(X_new, weighted_mean, weights, spherical=False))

                    #for full estimators
                    self.estimators_full_['means'].append(estimator_parms['means'][-1])
                    self.estimators_full_['covars'].append(estimator_parms['covars'][-1])

                    #<hyin/Feb-6th-2016> also remember the data weight according to the passive likelihood
                    #this could be useful if the weights according to the passive likelihood is desired for other applications
                    #to evaluate some statistics within the data parition
                    passive_likelihood_dict[e_idx, l_idx] = weights
                else:
                    estimator_parms['means'].append(np.mean(data_partition, axis=0))
                    estimator_parms['covars'].append(np.cov(data_partition.T))

                    #for full estimators
                    self.estimators_full_['means'].append(estimator_parms['means'][-1])
                    self.estimators_full_['covars'].append(estimator_parms['covars'][-1])

                    #for MaxEnt, uniform passive likelihood
                    passive_likelihood_dict[e_idx, l_idx] = np.ones(len(data_partition)) / float(len(data_partition))


                estimator_parms['weights'].append(weight_idx[e_idx, l_idx])
                self.estimators_full_['weights'].append(weight_idx[e_idx, l_idx]/float(self.n_estimators))

            self.estimators_.append(estimator_parms)
        #can stop here or go for expectation maximization for each estimator...
        if self.em_itrs > 0:
            #prepare em results for each estimator
            em_res = [self._em_steps(e_idx, X, y) for e_idx in range(self.n_estimators)]
            #or do EM on the full model?
            # <hyin/Dec-2nd-2015> no, doing this seems to harm the learning as the aggregated model is really
            # complex so optimizing that model tends to overfit...
            # em_res = self._em_steps(None, X, y)
            #then use them
            self.estimators_=em_res

        self.prepare_inv_and_constants()
        return indices, leaf_idx, partitioned_data, passive_likelihood_dict

    def _em_steps(self, estimator_idx, X, y=None):
        #use current estimation as initialization to perform expectation-maximization
        #now reuse the procedure implemented by scikit-learn, actually a costumized implementation
        #is required if the passive dynamics also needs to be learned.
        if self.verbose:
            if estimator_idx is not None:
                print 'EM steps for the estimator {0}'.format(estimator_idx)
            else:
                print 'EM steps...'

        if self.passive_dyn_func is not None and self.passive_dyn_ctrl is not None and self.passive_dyn_noise is not None:
            #extract X_old, X_new, X_new_passive
            X_old = X[:, 0:X.shape[1]/2]
            X_new = X[:, X.shape[1]/2:]
            X_new_passive = np.array([self.passive_dyn_func(X_old[sample_idx]) for sample_idx in range(X.shape[0])])


            # EM algorithms
            current_log_likelihood = None
            # reset self.converged_ to False
            converged = False
            # this line should be removed when 'thresh' is removed in v0.18
            tol = 1e-4
            #use the internal EM steps for non-uniform passive dynamics case
            for i in range(self.em_itrs):
                prev_log_likelihood = current_log_likelihood
                # Expectation step
                log_likelihoods, responsibilities = self._do_estep(
                    estimator_idx, X_new_passive, X_new, y)
                current_log_likelihood = log_likelihoods.mean()

                if self.verbose:
                    print 'current_log_likelihood:', current_log_likelihood
                if prev_log_likelihood is not None:
                    change = abs(current_log_likelihood - prev_log_likelihood)
                    if change < tol:
                        converged = True
                        break

                # Maximization step
                if estimator_idx is not None:
                    self._do_mstep(X_new_passive, X_new, responsibilities, self.estimators_[estimator_idx])
                else:
                    self._do_mstep(X_new_passive, X_new, responsibilities, self.estimators_full_)

            if estimator_idx is None:
                res=self.estimators_full_
            else:
                res=self.estimators_[estimator_idx]
        else:
            if estimator_idx is not None:
                n_partitions=len(self.estimators_[estimator_idx]['weights'])
                #use our own initialization
                g = mixture.GMM(n_components=n_partitions, n_iter=self.em_itrs, init_params='',
                    covariance_type='full')
                g.means_=np.array(self.estimators_[estimator_idx]['means'])
                g.covars_=np.array(self.estimators_[estimator_idx]['covars'])
                g.weights_=np.array(self.estimators_[estimator_idx]['weights'])
            else:
                n_partitions=len(self.estimators_full_['weights'])
                g = mixture.GMM(n_components=n_partitions, n_iter=self.em_itrs, init_params='',
                    covariance_type='full')
                g.means_=np.array(self.estimators_full_['means'])
                g.covars_=np.array(self.estimators_full_['covars'])
                g.weights_=np.array(self.estimators_full_['weights'])

            g.fit(X)

            #prepare to return a defaultdict
            res=defaultdict(list)
            res['means']=list(g.means_)
            res['covars']=list(g.covars_)
            res['weights']=list(g.weights_)

        return res

    def _do_estep(self, estimator_idx, X_new_passive, X_new, y):
        return self._score_sample_for_passive_mdl_helper(
                    estimator_idx, X_new_passive, X_new, y)

    def _do_mstep(self, X_new_passive, X_new, responsibilities, parms, min_covar=1e-7):
        """
        X_new_passive    -  An array of the propagation of the old state through the passiv edynamics
        X_new            -  An array of the new states that observed  
        responsibilities -  array_like, shape (n_samples, n_components)
                            Posterior probabilities of each mixture component for each data
        """
        n_samples, n_dim = X_new.shape
        weights = responsibilities.sum(axis=0)
        weighted_X_new_sum = np.dot(responsibilities.T, X_new)
        weighted_X_new_passive_sum = np.dot(responsibilities.T, X_new_passive)
        inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)
        weighted_X_new_mean = weighted_X_new_sum * inverse_weights
        weighted_X_new_passive_mean = weighted_X_new_passive_sum * inverse_weights

        if 'weights' in parms:
            parms['weights'] = (weights / (weights.sum() + 10 * EPS) + EPS)

        # delta_X_new                 = [None] * n_samples
        # delta_X_new_passive         = [None] * n_samples
        # delta_X_new_passive_Sigma_0 = [None] * n_samples
        # one_array = np.ones(n_dim)
        # for c in range(len(parms['weights'])):
        #     delta_X_new[c]                 = X_new - weighted_X_new_mean[c]
        #     delta_X_new_passive[c]         = X_new_passive - weighted_X_new_passive_mean[c]
        #     delta_X_new_passive_Sigma_0[c] = (1./self.passive_dyn_noise * np.eye(n_dim).dot(delta_X_new_passive[c].T)).T

        # if 'covars' in parms:
        #     #now only support diagonal covariance matrix
        #     for c, old_covar in enumerate(parms['covars']):
        #         constant=np.sum(delta_X_new[c]*delta_X_new[c]*responsibilities[:, c][:, np.newaxis], axis=0)#*inverse_weights[c, 0]
        #         so_coeff=np.sum(delta_X_new_passive_Sigma_0[c]*delta_X_new_passive_Sigma_0[c]*responsibilities[:, c][:, np.newaxis], axis=0)#*inverse_weights[c, 0]
        #         #take the roots for S matrix
        #         S_k=(np.sqrt(one_array+4*so_coeff*constant)-one_array)/(2*so_coeff)
        #         #get Sigma_k from S_k through S_k^(-1) = Sigma_k^(-1) + Sigma_0^(-1)
        #         Sigma_k = 1./(1./S_k -  1./self.passive_dyn_noise * np.ones(n_dim))
        #         print S_k, Sigma_k
        #         parms['covars'][c] = np.diag(Sigma_k)
        # if 'means' in parms:
        #     for c, old_mean in enumerate(parms['means']):
        #         Sigma_k_array = np.diag(parms['covars'][c])
        #         S_k=1./Sigma_k_array + 1./self.passive_dyn_noise * np.ones(n_dim)
        #         coeff_mat = np.diag(Sigma_k_array*(1./S_k))
        #         #difference betwen X_new and X_new_passive
        #         delta_X_new_X_new_passive = X_new - (np.diag(S_k).dot(X_new_passive.T)).T
        #         parms['means'][c] = coeff_mat.dot(np.sum(delta_X_new_X_new_passive*responsibilities[:, c][:, np.newaxis]*inverse_weights[c, 0], axis=0))
        #<hyin/Oct-23rd-2015> Try the formulation from Bellman equation, this seems leading t a weighted-linear regression problem...
        # c = (X_new - X_new_passive)
        #<hyin/OCt-27th-2015> Try the closed-form solutions for a relaxed lower-bound
        # if 'means' in parms:
        #     parms['means'] = weighted_X_new_mean
        # if 'covars' in parms:
        #     for c, old_covar in enumerate(parms['covars']):
        #         data_weights = responsibilities[:, c]
        #         parms['covars'][c] = _frequency_weighted_covariance(X_new, parms['means'][c], data_weights)

        #<hyin/Nov-20th-2015> As far as I realize, the above close-form solution actually optimize a value lower than the actual objective
        #however, this approximation is not tight thus unfortunately we cannot guarantee the optimum is also obtained for the actual objective...
        #another idea is to symplify the model by only learning the mean, or say the center of the RBF function
        #the width of the RBF basis can be adapted by solving a one-dimensional numerical optimization, this should lead to 
        #a generalized EM algorithm
        #<hyin/Jan-22nd-2016> note that without the adaptation of covariance, the shift of mean
        #is not that great option, so let's only keeps the weights adapatation. We need numerical optimization for the covariance adaptation
        #to see if it would help the mean shift 
        if 'means' in parms:
            for c, old_mean in enumerate(parms['means']):
                Sigma_k_array = parms['covars'][c]
                # S_k = self.passive_dyn_noise * self.passive_dyn_ctrl + Sigma_k_array + 1e-5*np.eye(X_new.shape[1])
                # # coeff_mat = np.diag(Sigma_k_array*(1./S_k))
                # inv_Sigma_k_array = np.linalg.pinv(Sigma_k_array)
                # inv_Sigma_sum = np.linalg.pinv(S_k + Sigma_k_array)
                # #could use woodbury here...
                # coeff_mat = np.linalg.pinv(inv_Sigma_k_array - inv_Sigma_sum)
                # #difference betwen X_new and X_new_passive
                # delta_X_new_X_new_passive = (inv_Sigma_k_array.dot(X_new.T) - inv_Sigma_sum.dot(X_new_passive.T)).T

                # parms['means'][c] = coeff_mat.dot(np.sum(delta_X_new_X_new_passive*responsibilities[:, c][:, np.newaxis]*inverse_weights[c, 0], axis=0))

                # #another formulation? which one is correct?
                # <hyin/Dec-2nd-2015> this seems more straightforward and at least give a keep increasing likelihood
                # need to check the original formulation to see whats the problem
                inv_Sigma_k_array = np.linalg.pinv(Sigma_k_array)
                inv_Sigma_0 = np.linalg.pinv(self.passive_dyn_noise * self.passive_dyn_ctrl + self.reg*np.eye(X_new.shape[1]))
                coeff_mat = Sigma_k_array
                inv_Sigma_sum = inv_Sigma_k_array + inv_Sigma_0
                delta_X_new_X_new_passive = (inv_Sigma_sum.dot(X_new.T) - inv_Sigma_0.dot(X_new_passive.T)).T
                parms['means'][c] = coeff_mat.dot(np.sum(delta_X_new_X_new_passive*responsibilities[:, c][:, np.newaxis]*inverse_weights[c, 0], axis=0))
        # return

    def sample(self, n_samples=1, random_state=None):
        '''
        return samples that are synthesized from the model
        '''
        if not hasattr(self, 'estimators_'):
            print 'The model has not been trained yet...'
            return
        else:
            pass
        return

    def score(self, X, y=None):
        #take log likelihood for each estimator for a given trajectory/state
        #without considering the passive dynamics: MaxEnt model
        estimator_scores=[_log_multivariate_normal_density_full(
                            X,
                            np.array(self.estimators_[e_idx]['means']),
                            np.array(self.estimators_[e_idx]['covars']))
                            +np.log(self.estimators_[e_idx]['weights']) for e_idx in range(self.n_estimators)]

        # concatenate different models...
        # estimator_scores=np.concatenate(estimator_scores,axis=1)
        # res=[logsumexp(x)-np.log(1./self.n_estimators) for x in np.array(estimator_scores)]
        # another way: mean of evaluated cost functions
        # helper to evaluate a single model
        mdl_eval = lambda scores: [logsumexp(x_score) for x_score in scores]
        estimator_scores = np.array([mdl_eval(scores) for scores in estimator_scores])

        responsibilities = [np.exp(estimator_scores[e_idx] - estimator_scores[e_idx][:, np.newaxis]) for e_idx in range(self.n_estimators)]
        #average seems to be more reasonable...
        res=np.mean(estimator_scores,axis=0)
        res_responsibilities = np.mean(np.array(responsibilities), axis=0)
        return -np.array(res), res_responsibilities

    def score_samples(self, X, y=None, min_covar=1.e-7):
        #a different version to evaluate the quality/likelihood of state pairs
        if self.passive_dyn_func is not None and self.passive_dyn_ctrl is not None and self.passive_dyn_noise is not None:
            X_old = X[:, 0:X.shape[1]/2]
            X_new = X[:, X.shape[1]/2:]
            X_new_passive = np.array([self.passive_dyn_func(X_old[sample_idx]) for sample_idx in range(X.shape[0])])

            log_prob_lst = [None] * self.n_estimators
            respon_lst = [None] * self.n_estimators
            for e_idx in range(self.n_estimators):
                log_prob_lst[e_idx], respon_lst[e_idx] = self._score_sample_for_passive_mdl_helper(
                    e_idx, X_new_passive, X_new, y, min_covar)
            res = -np.mean(np.array(log_prob_lst),axis=0)
            res_responsibilities = np.mean(np.array(respon_lst), axis=0)
        else:
            #this should be a trajectory/maximum ent model, use score...
            res, res_responsibilities = self.score(X, y)
        return res, res_responsibilities 


    def value_eval_samples(self, X, y=None, average=False, full=True, const=True):
        #switching off the constant term seems to smooth the value function
        #I don't quite understand why, my current guess is that the axis-align partition results in 
        #oversized covariance matrices, making the constant terms extremely large for some partitions
        #this can be shown adding a fixed term to the covariance matrices to mitigate the singularity
        #this could be cast as a kind of regularization

        #the new switch is actually equivalent to average=True, but since the training parameters are separated
        #lets keep this ugly solution...
        n_samples, n_dim = X.shape

        if not average:
            if not full:
                weights = []
                for idx in range(self.n_estimators):
                    weights = weights + (np.array(self.estimators_[idx]['weights'])/self.n_estimators).tolist()
                #the real function to evaluate the value functions, which are actually un-normalized Gaussians
                def value_estimator_eval(d):
                    res = []
                    for idx in range(self.n_estimators):
                        for i, (m, c_inv) in enumerate(   zip(self.estimators_[idx]['means'], 
                                                    self.estimators_[idx]['inv_covars'])):
                            diff_data = d - m
                            res.append(.5*diff_data.dot(c_inv).dot(diff_data) + self.estimators_[idx]['beta'][i]*const)
                    return np.array(res)

                res = np.array([ -logsumexp(-value_estimator_eval(d), b=np.array(weights)) for d in X])
            else:
                res = np.zeros(X.shape[0])
                res_mat = np.zeros((X.shape[0], len(self.estimators_full_['means'])))
                for i, (m, c_inv)   in enumerate(   zip(self.estimators_full_['means'], 
                                                self.estimators_full_['inv_covars'])):
                    diff_data = X - m
                    res_mat[:, i] = np.array([e_prod.dot(e)*0.5 + self.estimators_full_['beta'][i]*const for e_prod, e in zip(diff_data.dot(c_inv), diff_data)])
                for d_idx, r in enumerate(res_mat):
                    res[d_idx] = -logsumexp(-r, b=self.estimators_full_['weights'])
        else:
            #the real function to evaluate the value functions, which are actually un-normalized Gaussians
            def value_estimator_eval(idx):
                res = np.zeros((X.shape[0], len(self.estimators_[idx]['means'])))
                logsumexp_res=np.zeros(len(res))
                for i, (m, c_inv) in enumerate(   zip(self.estimators_[idx]['means'], 
                                            self.estimators_[idx]['inv_covars'])):
                    diff_data = X - m
                    res[:, i] = np.array([e_prod.dot(e)*0.5 + self.estimators_[idx]['beta'][i]*const for e_prod, e in zip(diff_data.dot(c_inv), diff_data)])
                for d_idx, r in enumerate(res):
                    logsumexp_res[d_idx] = -logsumexp(-r, b=self.estimators_[idx]['weights'])

                return logsumexp_res
                
            estimator_scores = [ value_estimator_eval(e_idx) for e_idx in range(self.n_estimators) ]
            #take average
            res = np.mean(np.array(estimator_scores), axis=0)
        return res
 
    def _score_sample_for_passive_mdl_helper(self, estimator_idx, X_new_passive, X_new, y, min_covar=1.e-7):
        #for the specified estimator with a passive dynamics model,
        #evaluate the likelihood for given state pairs
        #to call this, ensure passive dynamics and noise are available
        n_samples, n_dim = X_new.shape

        #incorporate the likelihood of passive dynamics - a Gaussian
        """
                        P_0(x'|x) exp^(V(x'))
        P(x'|x) = --------------------------------- = N(x', m(x), S)
                    int_x'' P_0(x''|x) exp^(V(x''))
        """
        """
        for sake of maximization step and simplicity, evaluate a lower-bound instead
        log(P(x'|x)) > -0.5 * D * log(2*pi) + 0.5*log(det(S^{-1})) -0.5*log2 + 0.5*log2 - 0.5*(x'-f(x))^TSigma^{-1}(x'-f(x)) - 0.5*(x'-mu_k)^TSimga_k^{-1}(x'-mu_k) + 0.5*(mu_k-f(x))^TM^{-1}(mu_k-f(x))
                     > -0.5 * D * log(2*pi) + 0.5*log(det(S^{-1})/2) + 0.5*log2 - 0.5*(x'-f(x))^TSigma^{-1}(x'-f(x)) - 0.5*(x'-mu_k)^TSimga_k^{-1}(x'-mu_k)
                     > -0.5 * D * log(2*pi) + 0.5*log((det(Sigma_k)^{-1}+det(Sigma_0)^{-1})/2) + 0.5*log2 - 0.5*(x'-f(x))^TSigma^{-1}(x'-f(x)) - 0.5*(x'-mu_k)^TSimga_k^{-1}(x'-mu_k) + 0.5*(mu_k-f(x))^TM^{-1}(mu_k-f(x))
                     > -0.5 * D * log(2*pi) + 0.5*log(det(Sigma_k)^{-1})/2 + 0.5*log(det(Sigma_0))/2 + 0.5*log2 - 0.5*(x'-f(x))^TSigma^{-1}(x'-f(x)) - 0.5*(x'-mu_k)^TSimga_k^{-1}(x'-mu_k) + 0.5*(mu_k-f(x))^TM^{-1}(mu_k-f(x))
        Any way to bound the last term to also make it independent from matrix other than Sigma_k?
        """

        # regularize to prevent numerical instability
        Sigma_0 = self.passive_dyn_noise * self.passive_dyn_ctrl + self.reg*np.eye(X_new.shape[1])
        # + 1e-2 * np.eye(X_new.shape[1])
        Sigma_0_inv = np.linalg.pinv(Sigma_0)
        if estimator_idx is not None:
            Sigma   = self.estimators_[estimator_idx]['covars']
            mu      = self.estimators_[estimator_idx]['means']
            w       = self.estimators_[estimator_idx]['weights']
        else:
            Sigma   = self.estimators_full_['covars']
            mu      = self.estimators_full_['means']
            w       = self.estimators_full_['weights']
        nmix    = len(mu)

        log_prob  = np.empty((n_samples, nmix))
        for c, (mu_k, Sigma_k) in enumerate(zip(mu, Sigma)):
            #obviously, this fraction can be optimized by exploiting the structure of covariance matrix
            #using say Cholesky decomposition
            Sigma_k_inv = np.linalg.pinv(Sigma_k)
            S_inv       = Sigma_k_inv + Sigma_0_inv
            S           = np.linalg.pinv(S_inv)
            try:
                S_chol = linalg.cholesky(S, lower=True)
            except linalg.LinAlgError:
                # The model is most probably stuck in a component with too
                # few observations, we need to reinitialize this components
                S_chol = linalg.cholesky(S + min_covar * np.eye(n_dim),
                                          lower=True)
            m = S.dot((Sigma_k_inv.dot(mu_k)+Sigma_0_inv.dot(X_new_passive.T).T).T).T
            #fraction part of above equation
            # scale_log_det = -.5 * (np.log(2*np.pi) + np.sum(np.log(S_inv)) + 
            #     2*np.sum(np.log(np.diag(Sigma_k_chol))) + np.sum(np.log(np.diag(Sigma_0))))
            # #exp() part of the above equation
            # S_sol = linalg.solve_triangular(M_chol, (X_new - X_old).T, lower=True).T

            # scale_log_rbf = -.5 * (np.sum(M_sol**2), axis=1)
            S_log_det = 2 * np.sum(np.log(np.diag(S_chol)))
            # print 'S_log_det:', S_log_det
            S_sol = linalg.solve_triangular(S_chol, (X_new - m).T, lower=True).T
            log_prob[:, c] = -.5 * (np.sum(S_sol**2, axis=1) + n_dim * np.log(2 * np.pi) + S_log_det)
        lpr = log_prob + np.log(w)
        # print 'log_prob:', log_prob
        # print 'w:', w
        # print 'lpr:', lpr
        logprob = logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])
        return logprob, responsibilities

    def prepare_inv_and_constants(self):
        '''
        supplement steps to prepare inverse of variance matrices and constant terms
        ''' 
        regularization = self.reg
        for idx in range(self.n_estimators):
            self.estimators_[idx]['inv_covars'] = [ np.linalg.pinv(covar + np.eye(covar.shape[0])*regularization) for covar in self.estimators_[idx]['covars']]
            self.estimators_[idx]['beta'] = [.5*np.log(pseudo_determinant(covar + np.eye(covar.shape[0])*regularization)) + .5*np.log(2*np.pi)*covar.shape[0] for covar in self.estimators_[idx]['covars']]

        self.estimators_full_['weights'] = []
        self.estimators_full_['means'] = []
        self.estimators_full_['covars'] = []
        for e_idx in range(self.n_estimators):
            for leaf_idx in range(len(self.estimators_[e_idx]['weights'])):
                self.estimators_full_['weights'].append(self.estimators_[e_idx]['weights'][leaf_idx]/float(self.n_estimators))
                self.estimators_full_['covars'].append(self.estimators_[e_idx]['covars'][leaf_idx])
                self.estimators_full_['means'].append(self.estimators_[e_idx]['means'][leaf_idx])
        # self.estimators_full_['inv_covars'] = [ np.linalg.pinv(covar) for covar in self.estimators_full_['covars']]
        # self.estimators_full_['beta'] = [.5*np.log(pseudo_determinant(covar)) + .5*np.log(2*np.pi)*covar.shape[0] for covar in self.estimators_full_['covars']]
                self.estimators_full_['inv_covars'].append(self.estimators_[e_idx]['inv_covars'][leaf_idx])
                self.estimators_full_['beta'].append(self.estimators_[e_idx]['beta'][leaf_idx])
        return

from scipy import linalg

def pseudo_determinant(S, thres=1e-3, min_covar=1.e-7):
    n_dim = S.shape[0]
    try:
        S_chol = linalg.cholesky(S, lower=True)
    except linalg.LinAlgError:
        # The model is most probably stuck in a component with too
        # few observations, we need to reinitialize this components
        S_chol = linalg.cholesky(S + min_covar * np.eye(n_dim),
                                  lower=True)
    S_chol_diag = np.diag(S_chol)

    return np.prod(S_chol_diag[S_chol_diag>thres]) ** 2

def _log_multivariate_normal_density_full(X, means, covars, min_covar=1.e-7):
    """
    Log probability for full covariance matrices.
    A shameless copy from scikit-learn
    """
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                      lower=True)
        cv_log_det = 2 * np.sum(np.log(np.diag(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob

def _passive_dyn_likelihood(X_new, X_new_passive, passive_dyn_noise, passive_dyn_ctrl, reg=1e-5):
    #regularized sigma
    sigma = passive_dyn_noise*passive_dyn_ctrl + reg*np.eye(X_new.shape[1])   
    #<hyin/Feb-9th-2016> slightly modify the sequence to prevent potential overflow issue
    denom = ((2*np.pi)**(X_new.shape[1]/2.0))*np.linalg.det(sigma)**.5
    err = X_new - X_new_passive
    err_prod = err.dot(np.linalg.pinv(sigma))
    quad_term = np.array([e.dot(ep) for e, ep in zip(err, err_prod)])
    
    num = np.exp(-.5*quad_term)
    return num/denom

def _frequency_weighted_covariance(X, m, weights, spherical=False):
    coeff = np.array(weights) / np.sum(weights)
    if spherical:
        #need numpy 1.9
        diff_data = np.linalg.norm(X - m, axis=1)
        sigma = np.sum(coeff * diff_data)
        covar = sigma * np.eye(len(m))
    else:
        diff_data = X - m
        covar = (coeff*diff_data.T).dot(diff_data)
        #need numpy 1.10
        # covar = np.cov(X, aweights=weights)

    return covar

def _stratified_weighted_covariance(X, m, weights):
    coeff = np.array(weights) / np.sum(weights)
    norm_coeff = 1./ (1. - np.sum(coeff**2))
    covar = np.zeros((X.shape[1], X.shape[1]))
    for j in range(covar.shape[0]):
        for k in range(covar.shape[1]):
            covar[j, k] = np.sum([c*(d[j]-m[j])*(d[k]-m[k]) for c, d in zip(coeff, X)])
    return covar

def EnsembleIOCTest():
    ''''
    A test to try modeling the occurences of state visiting
    Use the scikit-learn example
    '''
    n_samples = 300

    # generate random sample, two components
    np.random.seed(0)

    # generate spherical data centered on (20, 20)
    shifted_gaussian = 1.5*np.random.randn(n_samples, 2) + np.array([20, 20])

    # generate zero centered stretched Gaussian data
    C = np.array([[0., -0.7], [3.5, .7]])
    stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

    # concatenate the two datasets into the final training set
    X_train = np.vstack([shifted_gaussian, stretched_gaussian])

    model=EnsembleIOC(n_estimators=20, max_depth=3, min_samples_split=10, min_samples_leaf=10,
                        random_state=10,
                        em_itrs=15)
    #learn
    model.fit(X_train)
    # print len(model.estimators_)
    # print model.estimators_[0]['means']
    # print model.estimators_[0]['covars']
    # print model.estimators_[0]['weights']
    #visualize the data and heating map
    xmin=-20;xmax=30
    ymin=-20;ymax=40

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(X_train[:, 0], X_train[:, 1], .8)
    ax.hold(True)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    #evaluate testing points
    grid_dim1=np.linspace(xmin, xmax)
    grid_dim2=np.linspace(ymin, ymax)
    Sgrid=np.meshgrid(grid_dim1, grid_dim2)
    states = np.array([np.reshape(dim, (1, -1))[0] for dim in Sgrid])
    costs, _=model.score(states.T)
    pcol = ax.pcolormesh(grid_dim1, grid_dim2, 
        np.reshape(costs, (len(grid_dim1), len(grid_dim2))),
        shading='none')
    pcol.set_edgecolor('face')
    plt.show()

    return


if __name__ == '__main__':
    EnsembleIOCTest()