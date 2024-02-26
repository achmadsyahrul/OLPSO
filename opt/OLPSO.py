#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987

import numpy as np
from .tools import func_transformer
from .base import SkoBase
from .oa.oa_2levels import construct_OA
from .oa.evaluate_oed import evaluate_oed, evaluate_fa

class OLPSO(SkoBase):
    """
    Do OLPSO (Orthogonal Learning Particle swarm optimization) algorithm.

    This algorithm was adapted from the earlier works of Zhan, Zhi-Hui, Jun Zhang, and Ou Liu. 
    "Orthogonal learning particle swarm optimization."
    Proceedings of the 11th Annual conference on Genetic and evolutionary computation. 2009.

    The position update can be defined as:

    .. math::

       x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

    Where the position at the current step :math:`t` is updated using
    the computed velocity at :math:`t+1`. Furthermore, the velocity update
    is defined as:

    .. math::

       v_{ij}(t + 1) = w * v_{ij}(t) + cr_{j}(t)[po_{ij}(t) − x_{ij}(t)]

    Here, :math:`c` is the cognitive and social parameters. c is fixed to be 2.0.
    :math:`w` controls the inertia of the swarm's movement. w usually decreases
    linearly from 0.9 to 0.4 during the run time, so w = 0.9-0.5*gen/Generation.
    Using the OED method, the original PSO can be modified as an OLPSO with an
    OL strategy that combines information of Pi and Pn to form a better guidance vector Po.

    .. Zhan, Zhi-Hui, Jun Zhang, and Ou Liu. "Orthogonal learning particle swarm optimization."
    Proceedings of the 11th Annual conference on Genetic and evolutionary computation. 2009.

    Parameters
    --------------------
    func : function
        The func you want to do optimal
    dim : int
        Number of dimension, which is number of parameters of func.
    pop : int
        Size of population, which is the number of Particles. We use 'pop' to keep accordance with GA
    max_iter : int
        Max of iter iterations
    lb : array_like
        The lower bound of every variables of func
    ub : array_like
        The upper bound of every variables of func
    n_processes : int
        Number of processes, 0 means use all cpu
    X : ndarray
        Position of each particle
    V : ndarray
        Velocity of each particle
    G : int
        Number of reconstruction gap
    Attributes
    ----------------------
    pbest_x : array_like, shape is (pop,dim)
        best location of every particle in history
    pbest_y : array_like, shape is (pop,1)
        best image of every particle in history
    gbest_x : array_like, shape is (1,dim)
        general best location for all particles in history
    gbest_y : float
        general best image  for all particles in history
    gbest_x_hist : list
        gbest_x of every iteration
    gbest_y_hist : list
        gbest_y of every iteration

    """

    def __init__(self, func, n_dim=None, pop=40, max_iter=150, lb=-1e5, ub=1e5, w=0.9, c=2,
                 verbose=False, dim=None, n_processes=0, X=None, V=None, G=5):

        n_dim = n_dim or dim  # support the earlier version

        self.func = func_transformer(func, n_processes)
        self.w = w  # inertia
        self.c = c  # parameters to control personal best, global best respectively
        self.pop = pop  # number of particles
        self.n_dim = n_dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter
        self.verbose = verbose  # print the result of each iter or not
        self.G = G # gap to reconstruct po

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        if X is not None:
            self.X = X
        else:
            self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))

        if V is not None:
            self.V = V
        else:
            v_high = self.ub - self.lb
            self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.n_dim))  # speed of particles
        self.Y = self.cal_y()  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = np.array([[np.inf]] * pop)  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_x_hist = []
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()
        self.stagnated = np.array([[0]] * pop)

        self.OA = construct_OA(self.n_dim)
        self.po = self.pbest_x.copy()
        self.original_func = func # not transformed, not iterable purpose
        self.construct_po()

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.best_x, self.best_y = self.gbest_x, self.gbest_y  # history reasons, will be deprecated


    def construct_po(self):
        pn=self.gbest_x
        pn=pn.flatten()
        for i, pi in enumerate(self.pbest_x):
            min_val, min_combination, history, values = evaluate_oed(self.original_func, self.OA, pi, pn)
            fa_min_value, fa_min_combination = evaluate_fa(values, self.original_func, self.n_dim, pi, pn, min_val, min_combination)
            if fa_min_value < min_val:
                self.po[i] = fa_min_combination
            else:
                self.po[i] = min_combination
    
    def reconstruct_po(self, idx, pi):
        pn=self.gbest_x
        pn=pn.flatten()
        min_val, min_combination, history, values = evaluate_oed(self.original_func, self.OA, pi, pn)
        fa_min_value, fa_min_combination = evaluate_fa(values, self.original_func, self.n_dim, pi, pn, min_val, min_combination)
        if fa_min_value < min_val:
            self.po[idx] = fa_min_combination
        else:
            self.po[idx] = min_combination
    
    def update_V(self):
        r = np.random.rand(self.pop, self.n_dim)
        self.V = self.w * self.V + \
                 self.c * r * (self.po - self.X)

    def update_X(self):
        self.X = self.X + self.V
        self.X = np.clip(self.X, self.lb, self.ub)

    def cal_y(self):
        # calculate y for every x in X
        self.Y = self.func(self.X).reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y > self.Y
        for idx, x in enumerate(self.X):
            if self.need_update[idx]:
                self.stagnated[idx] = 0
            else:
                self.stagnated[idx] += 1
                if self.stagnated[idx] > self.G:
                    # print('{}. before {}'.format(idx, self.po))
                    self.reconstruct_po(idx, self.pbest_x[idx])
                    self.stagnated[idx] = 0
                    # print('after {}'.format(self.po))

        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)
    
    def run(self, max_iter=None, precision=None, N=20):
        '''
        precision: None or float
            If precision is None, it will run the number of max_iter steps
            If precision is a float, the loop will stop if continuous N difference between pbest less than precision
        N: int
        '''
        self.max_iter = max_iter or self.max_iter
        c = 0
        for iter_num in range(self.max_iter):
            self.w = 0.9 - 0.5*(iter_num/self.max_iter) #update inertia
            self.update_V()
            self.recorder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()
            if precision is not None:
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
                if tor_iter < precision:
                    c = c + 1
                    if c > N:
                        break
                else:
                    c = 0
            if self.verbose:
                print('Iter: {}, Best fit: {} at {}'.format(iter_num, self.gbest_y, self.gbest_x))

            self.gbest_x_hist.append(self.gbest_x)
            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

    fit = run

class OLPSOLocalBest(SkoBase):
    # Ring Topology
    def __init__(self, func, n_dim=None, pop=40, max_iter=150, lb=-1e5, ub=1e5, w=0.9, c=2,
                 verbose=False, dim=None, n_processes=0, X=None, V=None, G=5):

        n_dim = n_dim or dim  # support the earlier version

        self.func = func_transformer(func, n_processes)
        self.w = w  # inertia
        self.c = c  # parameters to control personal best, global best respectively
        self.pop = pop  # number of particles
        self.n_dim = n_dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter
        self.verbose = verbose  # print the result of each iter or not
        self.G = G # gap to reconstruct po
        self.neighbor_idx = np.zeros((self.pop, 2), dtype=int) # Ring Topology have 2 neighbors
        self.update_neighbor() # update neighbor

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        if X is not None:
            self.X = X
        else:
            self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))

        if V is not None:
            self.V = V
        else:
            v_high = self.ub - self.lb
            self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.n_dim))  # speed of particles
        self.Y = self.cal_y()  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = np.array([[np.inf]] * pop)  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_x_hist = []
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()
        self.stagnated = np.array([[0]] * pop)

        self.OA = construct_OA(self.n_dim)
        self.po = self.pbest_x.copy()
        self.original_func = func # not transformed, not iterable purpose
        self.construct_po()
        
        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.best_x, self.best_y = self.gbest_x, self.gbest_y  # history reasons, will be deprecated

    def update_neighbor(self):
    # Update neighbors idx, ring topology
        for i in range(self.pop):
            self.neighbor_idx[i][0] = (i - 1) % self.pop 
            self.neighbor_idx[i][1] = (i + 1) % self.pop
    
    def construct_po(self):
        pn=self.gbest_x
        pn=pn.flatten()
        for i, pi in enumerate(self.pbest_x):
            min_val, min_combination, history, values = evaluate_oed(self.original_func, self.OA, pi, pn)
            fa_min_value, fa_min_combination = evaluate_fa(values, self.original_func, self.n_dim, pi, pn, min_val, min_combination)
            if fa_min_value < min_val:
                self.po[i] = fa_min_combination
            else:
                self.po[i] = min_combination
    
    def reconstruct_po(self, idx, pi):
        pn=self.gbest_x
        pn=pn.flatten()
        min_val, min_combination, history, values = evaluate_oed(self.original_func, self.OA, pi, pn)
        fa_min_value, fa_min_combination = evaluate_fa(values, self.original_func, self.n_dim, pi, pn, min_val, min_combination)
        if fa_min_value < min_val:
            self.po[idx] = fa_min_combination
        else:
            self.po[idx] = min_combination

    def update_V(self):
        r = np.random.rand(self.pop, self.n_dim)
        self.V = self.w * self.V + \
                 self.c * r * (self.po - self.X)

    def update_X(self):
        self.X = self.X + self.V
        self.X = np.clip(self.X, self.lb, self.ub)

    def cal_y(self):
        # calculate y for every x in X
        self.Y = self.func(self.X).reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y > self.Y
        for idx, x in enumerate(self.X):
            if self.need_update[idx]:
                self.stagnated[idx] = 0
            else:
                self.stagnated[idx] += 1
                if self.stagnated[idx] > self.G:
                    # print('{}. before {}'.format(idx, self.po))
                    self.reconstruct_po(idx, self.pbest_x[idx])
                    self.stagnated[idx] = 0
                    # print('after {}'.format(self.po))

        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]
    
    def update_lbest(self):
        '''
        local best
        '''
        for i in range(self.pop):
            idx_min = np.argmin(self.pbest_y[self.neighbor_idx[i]])
            best_neighbor_idx = self.neighbor_idx[i][idx_min]
            if self.gbest_y > self.pbest_y[best_neighbor_idx]:
                self.gbest_x = self.pbest_x[best_neighbor_idx, :].copy()
                self.gbest_y = self.pbest_y[best_neighbor_idx]

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)

    def run(self, max_iter=None, precision=None, N=20):
        '''
        precision: None or float
            If precision is None, it will run the number of max_iter steps
            If precision is a float, the loop will stop if continuous N difference between pbest less than precision
        N: int
        '''
        self.max_iter = max_iter or self.max_iter
        c = 0
        for iter_num in range(self.max_iter):
            self.w = 0.9 - 0.5*(iter_num/self.max_iter) #update inertia
            self.update_V()
            self.recorder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_lbest()
            self.update_gbest()
            if precision is not None:
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
                if tor_iter < precision:
                    c = c + 1
                    if c > N:
                        break
                else:
                    c = 0
            if self.verbose:
                print('Iter: {}, Best fit: {} at {}'.format(iter_num, self.gbest_y, self.gbest_x))

            self.gbest_x_hist.append(self.gbest_x)
            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

    fit = run