import numpy as np
from opt.PSO import PSO, PSOLocalBest
from opt.OLPSO import OLPSO, OLPSOLocalBest
from opt.problems import rosenbrock

pop = 4 # size of population, number of particles
max_iter = 8 # max of iteration or max generation
lb = [-2, -2, -2] # lower bound position of every variables of func
ub = [2, 2, 2] # upper bound position of every variables of func
n_dim = 3 # number of dimension

np_lb, np_ub = np.array(lb) * np.ones(n_dim), np.array(ub) * np.ones(n_dim)
X = np.random.uniform(low=np_lb, high=np_ub, size=(pop, n_dim)) # position of each particle
v_high = np_ub - np_lb
V = np.random.uniform(low=-v_high, high=v_high, size=(pop, n_dim))  # velocity of each particle

# GPSO
gpso = PSO(func=rosenbrock, n_dim=n_dim, pop=pop, max_iter=max_iter, lb=lb, ub=ub, verbose=True, X=X, V=V)
gpso.record_mode = True
gpso.run()

# LPSO, Ring Topology
lpso = PSOLocalBest(func=rosenbrock, n_dim=n_dim, pop=pop, max_iter=max_iter, lb=lb, ub=ub, verbose=True, X=X, V=V)
lpso.record_mode = True
lpso.run()

# OLPSO-G
olpso = OLPSO(func=rosenbrock, n_dim=n_dim, pop=pop, max_iter=max_iter, lb=lb, ub=ub, verbose=True, X=X, V=V)
olpso.record_mode = True
olpso.run()

# OLPSO-L, Ring Topology
olpso_l = OLPSOLocalBest(func=rosenbrock, n_dim=n_dim, pop=pop, max_iter=max_iter, lb=lb, ub=ub, verbose=True, X=X, V=V)
olpso_l.record_mode = True
olpso_l.run()