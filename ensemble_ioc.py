"""
A module that implements the ensemble of inverse optimal control models
"""
import cPickle as cp
from collections import defaultdict
import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn import decomposition
from sklearn.ensemble import RandomTreesEmbedding, RandomForestRegressor
from sklearn.linear_model import LinearRegression
# <hyin/Oct-23rd replace sklearn GMM as it starts deprecating since 0.18
# from sklearn import mixture
import gmr.gmr.gmm as gmm
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN

import scipy.optimize as sciopt
import scipy.stats as sps
from scipy.misc import logsumexp

EPS = np.finfo(float).eps

class EnsembleIOC(BaseEstimator, RegressorMixin):
    '''
    Handling state/state pairs as input
    '''
    def __init__(self,  n_estimators=20,
                        max_depth=5, min_samples_split=10, min_samples_leaf=10, clustering=0,
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
        clustering          - whether or not to force the number of subset. If non-zero, call a clustering scheme with the learned metric
        em_itrs             - maximum number of EM iterations to take if one would like to increase the likelihood of the MaxEnt approximation
        regularization      - small positive scalar to prevent singularity of matrix inversion. This is especially necessary when passive dynamics
                              is considered. Notably, the underactuated system will assum zero covariance for uncontrolled state dimensions but this might not
                              not be the case in reality since the collected data could be corrupted by noises.
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
        self.clustering=clustering
        self.random_state=random_state
        self.em_itrs=em_itrs
        self.reg=regularization
        self.passive_dyn_func=passive_dyn_func
        self.passive_dyn_ctrl=passive_dyn_ctrl
        self.passive_dyn_noise=passive_dyn_noise
        self.verbose=verbose
        return

    def predict(self, X):
        n_samples, n_dim = X.shape

        # use approximated GMM to capture the correlation, which provides us an initialization to iterate
        # the MAP estimation
        tmp_gmm = gmm.GMM(  n_components=len(self.gmm_estimators_full_['weights']),
                            priors=np.array(self.gmm_estimators_full_['weights']),
                            means=np.array(self.gmm_estimators_full_['means']),
                            covariances=self.gmm_estimators_full_['covars'])

        init_guess, init_covar = tmp_gmm.predict_with_covariance(indices=range(n_dim), X=X)

        def objfunc(x, *args):
            prior_mu, prior_inv_var = args
            vals, grads = self.value_eval_samples_helper(np.array([x]), average=False, const=True)
            prior_prob = .5*(x - prior_mu).dot(prior_inv_var).dot(x - prior_mu)
            prior_grad = prior_inv_var.dot(x-prior_mu)
            return vals[0] + prior_prob, grads[0] + prior_grad

        res = []
        for sample_idx in range(n_samples):
            opt_res = sciopt.minimize(  fun=objfunc,
                                        x0=init_guess[sample_idx, :],
                                        args=(init_guess[sample_idx, :], np.linalg.pinv(init_covar[sample_idx])),
                                        method='BFGS',
                                        jac=True,
                                        options={'gtol': 1e-8, 'disp': False})
            # print opt_res.message, opt_res.x,
            # print opt_res.fun, opt_res.jac
            # print init_guess[sample_idx, :], init_covar[sample_idx], opt_res.x
            res.append(opt_res.x)
        res = np.array(res)
        return res

    def _check_grads(self, X):
        n_samples, n_dim = X.shape

        # #predict the next state x_{t+1} given x_{t}
        tmp_gmm = gmm.GMM(  n_components=len(self.gmm_estimators_full_['weights']),
                            priors=np.array(self.gmm_estimators_full_['weights']),
                            means=np.array(self.gmm_estimators_full_['means']),
                            covariances=self.gmm_estimators_full_['covars'])

        init_guess, init_covar = tmp_gmm.predict_with_covariance(indices=range(n_dim), X=X)

        def objfunc(x, *args):
            prior_mu, prior_var = args
            vals, grads = self.value_eval_samples_helper(np.array([x]), average=False, const=True)
            prior_prob = .5*(x - prior_mu).dot(prior_var).dot(x - prior_mu)
            prior_grad = prior_var.dot(x-prior_mu)
            return vals[0] + prior_prob, grads[0] + prior_grad

        res = []
        for sample_idx in range(n_samples):
            def check_grad_fun(x):
                return objfunc(x, init_guess[sample_idx, :], init_covar[sample_idx])[0]
            def check_grad_fun_jac(x):
                return objfunc(x, init_guess[sample_idx, :], init_covar[sample_idx])[1]

            res.append(sciopt.check_grad(check_grad_fun, check_grad_fun_jac, X[sample_idx, :]))

        return np.mean(res)

    def fit(self, X, y=None):
        '''
        X - an array of concatenated features X_i = (x_{t-1}, x_{t}) corresponding to the infinite horizon case
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

        n_samples, n_dims = X.shape

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
            # self.random_embedding_mdl_.fit(X[:, X.shape[1]/2:])
            # indices = self.random_embedding_mdl_.apply(X[:, X.shape[1]/2:])
            self.random_embedding_mdl_.fit(X[:, :X.shape[1]/2])
            indices = self.random_embedding_mdl_.apply(X[:, :X.shape[1]/2])
            # X_tmp = np.array(X)
            # X_tmp[:, X.shape[1]/2:] = X_tmp[:, X.shape[1]/2:] - X_tmp[:, :X.shape[1]/2]
            # self.random_embedding_mdl_.fit(X_tmp)

            # indices = self.random_embedding_mdl_.apply(X_tmp)
        else:
            self.random_embedding_mdl_.fit(X)
            #figure out indices
            indices = self.random_embedding_mdl_.apply(X)

        #prepare ensemble for prediction
        self.random_prediction_mdl_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
            )

        self.random_prediction_mdl_.fit(X[:, :X.shape[1]/2], X[:, X.shape[1]/2:])

        if self.clustering > 0:
            #we need to force the data to situate in clusters with the given number and the random embeddings
            #first construct affinity
            #use extracted indices as sparse features to construct an affinity matrix
            if self.n_estimators > 1:
                if self.verbose:
                    print 'Building {0} subset of data depending on their random embedding similarity...'.format(self.clustering)
                #it makes sense to use the random embedding to do the clustering if we have ensembled features
                aff_mat = _affinity_matrix_from_indices(indices, 'binary')
                #using spectral mapping (Laplacian eigenmap)
                self.cluster = SpectralClustering(n_clusters=self.clustering, affinity='precomputed')
                self.cluster.fit(aff_mat)
            else:
                if self.verbose:
                    print 'Building {0} subset of data depending on their Euclidean similarity...'.format(self.clustering)
                #otherwise, use euclidean distance, this should be enough when the state space is low dimensional
                self.cluster = KMeans(n_clusters=self.clustering, max_iter=200, n_init=5)
                self.cluster.fit(X)

            partitioned_data = defaultdict(list)
            leaf_idx = defaultdict(set)
            weight_idx = defaultdict(float)
            for d_idx, d, p_idx in zip(range(len(X)), X, self.cluster.labels_):
                partitioned_data[0, p_idx].append(d)
                leaf_idx[0] |= {p_idx}
            for p_idx in range(self.clustering):
                weight_idx[0, p_idx] = 1./self.clustering
            num_estimators = 1
        else:
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
            num_estimators = self.n_estimators

        #for each grouped data, solve an easy IOC problem by assuming quadratic cost-to-go function
        #note that, if the passive dynamics need to be learned, extra steps is needed to train a regressor with weighted data
        #otherwise, just a simply gaussian for each conditional probability distribution model
        self.estimators_ = []
        #another copy to store the parameters all together, for EM/evaluation on all of the models
        self.estimators_full_ = defaultdict(list)

        #<hyin/Feb-6th-2016> an estimator and leaf indexed structure to record the passive likelihood of data...
        passive_likelihood_dict = defaultdict(list)
        for e_idx in range(num_estimators):
            #for each estimator
            estimator_parms = defaultdict(list)
            for l_idx in leaf_idx[e_idx]:
                if self.verbose:
                    print 'Processing {0}-th estimator and {1}-th leaf/partition...'.format(e_idx, l_idx)
                #and for each data partition
                data_partition=np.array(partitioned_data[e_idx, l_idx])

                estimator_parms['means'].append(np.mean(data_partition, axis=0))
                estimator_parms['covars'].append(np.cov(data_partition.T) + np.eye(data_partition.shape[1])*self.reg)

                #for MaxEnt, uniform passive likelihood
                passive_likelihood_dict[e_idx, l_idx] = np.ones(len(data_partition)) / float(len(data_partition))


                estimator_parms['weights'].append(weight_idx[e_idx, l_idx])

            self.estimators_.append(estimator_parms)

        #can stop here or go for expectation maximization for each estimator...
        if self.em_itrs > 0:
            #prepare em results for each estimator
            em_res = [self._em_steps(e_idx, X, y) for e_idx in range(num_estimators)]

            self.estimators_ = em_res

        #record the gmm approximation
        self.gmm_estimators_ = copy.deepcopy(self.estimators_)
        self.gmm_estimators_full_ = defaultdict(list)

        for est in self.estimators_:
            for comp_idx in range(len(est['weights'])):
                est['means'][comp_idx] = est['means'][comp_idx][(n_dims/2):]
                est['covars'][comp_idx] = est['covars'][comp_idx][(n_dims/2):, (n_dims/2):]
                self.estimators_full_['weights'].append(est['weights'][comp_idx]/float(num_estimators))
                #for full estimators
                self.estimators_full_['means'].append(est['means'][comp_idx])
                self.estimators_full_['covars'].append(est['covars'][comp_idx])

        if self.passive_dyn_func is not None and self.passive_dyn_ctrl is not None and self.passive_dyn_noise is not None:
            X_new         = X[:, X.shape[1]/2:]
            X_old         = X[:, 0:X.shape[1]/2]

            #merge the model knowledge if passive dynamics model is available, use MaxEnt assumption otherwise
            X_new_passive = np.array([self.passive_dyn_func(X_old[sample_idx]) for sample_idx in range(X.shape[0])])
            passive_likelihood = _passive_dyn_likelihood(X_new, X_new_passive, self.passive_dyn_noise, self.passive_dyn_ctrl, self.reg)
            weights = passive_likelihood / (np.sum(passive_likelihood) + self.reg)

            if np.sum(weights) < 1e-10:
                weights = 1./len(weights) * np.ones(len(weights))
            #a GMM as a MaxEnt surrogate
            tmp_gmm = gmm.GMM(  n_components=len(self.estimators_[0]['weights']),
                                priors=self.estimators_[0]['weights'],
                                means=self.estimators_[0]['means'],
                                covariances=self.estimators_[0]['covars'])
            for e_idx in range(num_estimators):
                tmp_gmm.n_components = len(self.estimators_[e_idx]['weights'])
                tmp_gmm.priors = self.estimators_[e_idx]['weights']
                tmp_gmm.means = self.estimators_[e_idx]['means']
                tmp_gmm.covariances = self.estimators_[e_idx]['covars']

                responsibilities = tmp_gmm.to_responsibilities(X_new)
                responsibilities = responsibilities / (np.sum(responsibilities, axis=0) + 1e-10)
                new_weights = (weights * responsibilities.T).T

                new_weights = (new_weights + 1e-10) / (np.sum(new_weights +1e-10, axis=0))

                weighted_means = [np.sum((new_weight*X_new.T).T, axis=0) for new_weight in new_weights.T]

                weighted_covars =[ _frequency_weighted_covariance(X_new, weighted_mean, new_weight, spherical=False)
                                        for new_weight, weighted_mean in zip(new_weights.T, weighted_means)]

                self.estimators_[e_idx]['means'] = weighted_means
                self.estimators_[e_idx]['covars'] = weighted_covars


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

        if estimator_idx is not None:
            n_partitions=len(self.estimators_[estimator_idx]['weights'])
            if self.verbose:
                print 'num of partitions:', n_partitions
            #use our own initialization
            g = gmm.GMM(n_components=n_partitions, priors=np.array(self.estimators_[estimator_idx]['weights']),
                means=np.array(self.estimators_[estimator_idx]['means']),
                covariances=np.array(self.estimators_[estimator_idx]['covars']),
                n_iter=self.em_itrs,
                covariance_type='full')
        else:
            n_partitions=len(self.estimators_full_['weights'])
            g = mixture.GaussianMixture(n_components=n_partitions, priors=np.array(self.estimators_[estimator_idx]['weights']),
                means=np.array(self.estimators_[estimator_idx]['means']),
                covariances=np.array(self.estimators_[estimator_idx]['covars']),
                n_iter=self.em_itrs,
                covariance_type='full')

        # g.fit(X[:, (X.shape[1]/2):])
        g.fit(X)

        #prepare to return a defaultdict
        res=defaultdict(list)
        res['means']=list(g.means)
        res['covars']=list(g.covariances)
        res['weights']=list(g.priors)

        return res

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
        return self.value_eval_samples(X, y, False, True)

    def value_eval_samples(self, X, y=None, average=False, const=True):
        scores, grads = self.value_eval_samples_helper(X, y, average, const)
        return scores

    def value_eval_samples_helper(self, X, y=None, average=False, const=True):
        n_samples, n_dim = X.shape

        grads = np.zeros((n_samples, n_dim))

        if self.clustering > 0:
            num_estimators = 1
        else:
            num_estimators = self.n_estimators

        if not average:
            res = np.zeros(X.shape[0])
            res_mat = np.zeros((X.shape[0], len(self.estimators_full_['means'])))
            res_grad_tmp = []
            for i, (m, c_inv)   in enumerate(   zip(self.estimators_full_['means'],
                                            self.estimators_full_['inv_covars'])):
                diff_data = X - m
                res_mat[:, i] = np.array([e_prod.dot(e)*0.5 + self.estimators_full_['beta'][i]*const for e_prod, e in zip(diff_data.dot(c_inv), diff_data)])
                res_grad_tmp.append(c_inv.dot(diff_data.T).T)
            for d_idx, r in enumerate(res_mat):
                res[d_idx] = -logsumexp(-r, b=np.array(self.estimators_full_['weights']))
            resp = ((np.exp(-res_mat)*np.array(self.estimators_full_['weights'])).T / np.exp(-res)).T
            for e_idx in range(res_mat.shape[1]):
                grads += (res_grad_tmp[e_idx].T * resp[:, e_idx]).T
        else:
            def value_estimator_eval(d, est_idx):
                res = []
                for i, (m, c_inv) in enumerate(   zip(self.estimators_[est_idx]['means'],
                                            self.estimators_[est_idx]['inv_covars'])):
                    diff_data = d - m
                    res.append((.5*diff_data.dot(c_inv).dot(diff_data.T) + self.estimators_[est_idx]['beta'][i]*const)[0])
                return np.array(res).T
            def value_estimator_grad(d, est_idx, val):
                res_grad = 0
                for i, (m, c_inv) in enumerate(   zip(self.estimators_[est_idx]['means'],
                                            self.estimators_[est_idx]['inv_covars'])):
                    diff_data = d - m
                    resp = np.exp(-(.5*diff_data.dot(c_inv).dot(diff_data.T) + self.estimators_[est_idx]['beta'][i]*const)[0]) * self.estimators_[est_idx]['weights'][i]
                    grad_comp = c_inv.dot(diff_data.T).T
                    res_grad += (grad_comp.T * (resp / np.exp(-val))).T
                return res_grad
            res = np.array([-logsumexp(-value_estimator_eval(X, idx), axis=1, b=self.estimators_[idx]['weights']) for idx in range(num_estimators)]).T
            res_grad = [value_estimator_grad(X, idx, res[:, idx]) for idx in range(num_estimators)]
            res = np.mean(res, axis=1)
            grads = np.mean(res_grad, axis=0)
        return res, grads

    def prepare_inv_and_constants(self):
        '''
        supplement steps to prepare inverse of variance matrices and constant terms
        '''
        regularization = self.reg

        if self.clustering > 0:
            num_estimators = 1
        else:
            num_estimators = self.n_estimators

        for idx in range(num_estimators):
            self.estimators_[idx]['inv_covars'] = [ np.linalg.pinv(covar + np.eye(covar.shape[0])*regularization) for covar in self.estimators_[idx]['covars']]
            self.estimators_[idx]['beta'] = [.5*np.log(pseudo_determinant(covar + np.eye(covar.shape[0])*regularization)) + .5*np.log(2*np.pi)*covar.shape[0] for covar in self.estimators_[idx]['covars']]

        self.estimators_full_['weights'] = []
        self.estimators_full_['means'] = []
        self.estimators_full_['covars'] = []

        self.gmm_estimators_full_['weights'] = []
        self.gmm_estimators_full_['means'] = []
        self.gmm_estimators_full_['covars'] = []
        for e_idx in range(num_estimators):
            for leaf_idx in range(len(self.estimators_[e_idx]['weights'])):
                self.estimators_full_['weights'].append(self.estimators_[e_idx]['weights'][leaf_idx]/float(num_estimators))
                self.estimators_full_['covars'].append(self.estimators_[e_idx]['covars'][leaf_idx])
                self.estimators_full_['means'].append(self.estimators_[e_idx]['means'][leaf_idx])

                self.estimators_full_['inv_covars'].append(self.estimators_[e_idx]['inv_covars'][leaf_idx])
                self.estimators_full_['beta'].append(self.estimators_[e_idx]['beta'][leaf_idx])

                self.gmm_estimators_full_['weights'].append(self.gmm_estimators_[e_idx]['weights'][leaf_idx]/float(num_estimators))
                self.gmm_estimators_full_['covars'].append(self.gmm_estimators_[e_idx]['covars'][leaf_idx])
                self.gmm_estimators_full_['means'].append(self.gmm_estimators_[e_idx]['means'][leaf_idx])
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

def _passive_dyn_likelihood_helper(X_new, X_new_passive, passive_dyn_noise, passive_dyn_ctrl, reg=1e-5):
    #regularized sigma
    log_grads = np.zeros(X_new.shape)
    sigma = passive_dyn_noise*passive_dyn_ctrl + reg*np.eye(X_new.shape[1])
    #<hyin/Feb-9th-2016> slightly modify the sequence to prevent potential overflow issue
    denom = ((2*np.pi)**(X_new.shape[1]/2.0))*np.linalg.det(sigma)**.5
    err = X_new - X_new_passive
    err_prod = err.dot(np.linalg.pinv(sigma))
    quad_term = np.array([e.dot(ep) for e, ep in zip(err, err_prod)])
    num = np.exp(-.5*quad_term)
    log_likelihood = -.5*quad_term - np.log(denom)
    log_grads = -err_prod
    return num/denom, log_likelihood, log_grads
def _passive_dyn_likelihood(X_new, X_new_passive, passive_dyn_noise, passive_dyn_ctrl, reg=1e-5):
    likelihoods, _, _ = _passive_dyn_likelihood_helper(X_new, X_new_passive, passive_dyn_noise, passive_dyn_ctrl, reg)
    return likelihoods

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

def _affinity_matrix_from_indices(indices, metric='binary', param=1.0):
    #input is an array of data represented by sparse encoding
    if metric == 'binary':
        #binary metric is parm free
        aff_op = lambda a, b: np.mean([int(ind_a==ind_b) for ind_a, ind_b in zip(a, b)])
    elif metric == 'gaussian':
        aff_op = lambda a, b: np.mean([np.exp(-(a-b).dot(a-b)*param) if ind_a==ind_b else 0 for ind_a, ind_b in zip(a, b)])
    elif metric == 'mahalanobis':
        aff_op = lambda a, b: np.mean([np.exp(-param.dot(a-b).dot(a-b)) if ind_a==ind_b else 0 for ind_a, ind_b in zip(a, b)])
    else:
        aff_op = None

    if aff_op is not None:
        n_samples = indices.shape[0]
        aff_mat = [[aff_op(indices[i], indices[j]) for j in range(n_samples)] for i in range(n_samples)]
    else:
        print 'Invalid metric specified.'
        aff_mat = None
    return aff_mat

class EnsembleIOCTraj(BaseEstimator, RegressorMixin):
    '''
    Handling the entire trajectories as the input
    '''
    def __init__(self,  traj_clusters=3, ti=True,
                        n_estimators=20,
                        max_depth=5, min_samples_split=10, min_samples_leaf=10, state_n_estimators=100, state_n_clusters=0,
                        random_state=0,
                        em_itrs=5,
                        regularization=0.05,
                        passive_dyn_func=None,
                        passive_dyn_ctrl=None,
                        passive_dyn_noise=None,
                        verbose=False):
        '''
        traj_clusters       - number of clusters of trajectories
        ti                  - whether or not to extract time invariant states

        ***The remained parameters are for the state ioc estimators***
        n_estimators        - number of ensembled models
        ...                 - a batch of parameters used for RandomTreesEmbedding, see relevant documents

        state_n_estimators  - number of state estimators
        state_n_clusters    - number of clusters for states for each trajectory group
        em_itrs             - maximum number of EM iterations to take
        regularization      - small positive scalar to prevent singularity of matrix inversion
        passive_dyn_func    - function to evaluate passive dynamics; None for MaxEnt model
        passive_dyn_ctrl    - function to return the control matrix which might depend on the state...
        passive_dyn_noise   - covariance of a Gaussian noise; only applicable when passive_dyn is Gaussian; None for MaxEnt model
                                note this implies a dynamical system with constant input gain. It is extendable to have state dependent
                                input gain then we need covariance for each data point
        verbose             - output training information
        '''
        self.n_traj_clusters = traj_clusters
        if isinstance(state_n_clusters, int):
            state_clusters_lst = [state_n_clusters] * self.n_traj_clusters
        else:
            state_clusters_lst = state_n_clusters

        self.eioc_mdls = [ EnsembleIOC( n_estimators=state_n_estimators,
                                        max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, clustering=state_n_clusters,  #let random embedding decides how many clusters we should have
                                        random_state=random_state,
                                        em_itrs=em_itrs,
                                        regularization=regularization,
                                        passive_dyn_func=passive_dyn_func,
                                        passive_dyn_ctrl=passive_dyn_ctrl,
                                        passive_dyn_noise=passive_dyn_noise,
                                        verbose=verbose) for i in range(self.n_traj_clusters) ]
        self.ti = ti
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf
        self.random_state=random_state
        self.state_n_estimators = state_n_estimators
        self.state_n_clusters = state_n_clusters
        self.em_itrs=em_itrs
        self.reg=regularization
        self.passive_dyn_func=passive_dyn_func
        self.passive_dyn_ctrl=passive_dyn_ctrl
        self.passive_dyn_noise=passive_dyn_noise
        self.verbose=verbose

        self.clustered_trajs = None
        return

    def cluster_trajectories(self, trajs):
        #clustering the trajectories according to random embedding parameters and number of clusters
        #flatten each trajectories
        flattened_trajs = np.array([np.array(traj).T.flatten() for traj in trajs])

        #an initial partitioning of data with random forest embedding
        self.random_embedding_mdl_ = RandomTreesEmbedding(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
            )

        self.random_embedding_mdl_.fit(flattened_trajs)
        #figure out indices
        indices = self.random_embedding_mdl_.apply(flattened_trajs)

        #we need to force the data to situate in clusters with the given number and the random embeddings
        #first construct affinity
        #use extracted indices as sparse features to construct an affinity matrix
        if self.verbose:
            print 'Building {0} subset of trajectories depending on their random embedding similarity...'.format(self.n_traj_clusters)
        aff_mat = _affinity_matrix_from_indices(indices, 'binary')
        #using spectral mapping (Laplacian eigenmap)
        self.cluster = SpectralClustering(n_clusters=self.n_traj_clusters, affinity='precomputed')
        self.cluster.fit(aff_mat)

        clustered_trajs = [[] for i in range(self.n_traj_clusters)]

        for d_idx, d, p_idx in zip(range(len(trajs)), trajs, self.cluster.labels_):
            clustered_trajs[p_idx].append(d)

        #let's see how the DBSCAN works
        #here it means at least how many trajectories do we need to form a cluster
        #dont know why always assign all of the data as noise...
        # self.cluster = DBSCAN(eps=0.5, min_samples=self.n_traj_clusters, metric='euclidean', algorithm='auto')
        # flatten_trajs = [traj.T.flatten() for traj in trajs]
        # self.cluster.fit(flatten_trajs)
        # labels = self.cluster.labels_
        # print labels
        # # Number of clusters in labels, ignoring noise if present.
        # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        #
        # clustered_trajs = [[] for i in range(n_clusters_)]
        #
        # for d_idx, d, p_idx in zip(range(len(trajs)), trajs, labels):
        #     clustered_trajs[p_idx].append(d)

        return np.array(clustered_trajs)

    def fit(self, X, y=None):
        '''
        X is an array of trajectories
        '''
        #first cluster these trajectories to locally similar data sets (here 'locally' does not necessarily mean euclidean distance)
        clustered_trajs = self.cluster_trajectories(X)

        for i in range(len(clustered_trajs)):
            #for each clustered trajectories train the sub eioc model
            #reform the trajectories if necessary
            if not self.ti:
                #time varing system, just flatten them
                flattened_trajs = [ np.array(traj).T.flatten() in clustered_trajs[i]]
                self.eioc_mdls[i].clustering=1
                self.eioc_mdls[i].fit(flattened_trajs)
                #note the fit model retains mean and covariance of the flattened trajectories
            else:
                #time invariant
                aug_states = []
                for traj in clustered_trajs[i]:
                    for t_idx in range(len(traj)-1):
                        aug_states.append(np.array(traj)[t_idx:t_idx+2, :].flatten())

                self.eioc_mdls[i].fit(np.array(aug_states))

        self.clustered_trajs = clustered_trajs
        return

    def score(self, X, gamma=1.0, average=False):
        #score a query state
        if self.clustered_trajs is not None:
            #the model ensemble has been trained
            # score_ensemble = [np.array(model.score(X)[0]) for model in self.eioc_mdls]
            score_ensemble = [np.array(model.value_eval_samples(X,average=average)) for model in self.eioc_mdls]
            #average (maximum likelihood) or logsumexp (softmaximum -> maximum posterior)
            if gamma is None:
                res = np.mean(score_ensemble, axis=0)
            else:
                # mdl_eval = lambda scores: [logsumexp(x_score) for x_score in scores]
                res = np.array([-logsumexp(-gamma*np.array([score[sample_idx] for score in score_ensemble])) for sample_idx, sample in enumerate(X)])

        return res

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
    np.random.shuffle(X_train)

    model=EnsembleIOC(n_estimators=1, max_depth=3, min_samples_split=10, min_samples_leaf=10, clustering=2,
                        random_state=10,
                        em_itrs=0)
    #learn
    indices, leaf_idx, partitioned_data, passive_likelihood_dict = model.fit(X_train)
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

    colors = ['b','w']
    for idx, c in enumerate(colors):
        pnts = np.array(partitioned_data[0, idx])
        ax.plot(pnts[:, 0], pnts[:, 1], '*', color=c)
        mean = model.estimators_[0]['means'][idx]
        ax.plot([mean[0]], [mean[1]], 'o', markersize=24, color=c)
        print mean, model.estimators_[0]['covars'][idx]
    plt.show()

    return


if __name__ == '__main__':
    EnsembleIOCTest()
