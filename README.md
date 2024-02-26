# Orthogonal Learning Particle Swarm Optimization (OLPSO)

## Overview
Orthogonal Learning Particle Swarm Optimization (OLPSO) is a particle swarm optimization algorithm that uses an orthogonal learning strategy to improve PSO performance.

## Usage / Example
-> Demo code: [example_usage.py](https://github.com/achmadsyahrul/OLPSO/blob/master/example_usage.py)

### OLPSO
```python
import numpy as np
from opt.OLPSO import OLPSO, OLPSOLocalBest
from opt.problems import rosenbrock

# Define parameters and initialize population
pop = 4 # size of population, number of particles
max_iter = 8 # max of iteration or max generation
lb = [-2, -2, -2] # lower bound position of every variables of func
ub = [2, 2, 2] # upper bound position of every variables of func
n_dim = 3 # number of dimension

np_lb, np_ub = np.array(lb) * np.ones(n_dim), np.array(ub) * np.ones(n_dim)
X = np.random.uniform(low=np_lb, high=np_ub, size=(pop, n_dim)) # position of each particle
v_high = np_ub - np_lb
V = np.random.uniform(low=-v_high, high=v_high, size=(pop, n_dim))  # velocity of each particle

# OLPSO-G
olpso = OLPSO(func=rosenbrock, n_dim=n_dim, pop=pop, max_iter=max_iter, lb=lb, ub=ub, verbose=True, X=X, V=V)
olpso.record_mode = True
olpso.run()

# OLPSO-L, Ring Topology
olpso_l = OLPSOLocalBest(func=rosenbrock, n_dim=n_dim, pop=pop, max_iter=max_iter, lb=lb, ub=ub, verbose=True, X=X, V=V)
olpso_l.record_mode = True
olpso_l.run()
```

### PSO
If you want compare with PSO
```python
from opt.PSO import PSO, PSOLocalBest

# GPSO
gpso = PSO(func=rosenbrock, n_dim=n_dim, pop=pop, max_iter=max_iter, lb=lb, ub=ub, verbose=True, X=X, V=V)
gpso.record_mode = True
gpso.run()

# LPSO, Ring Topology
lpso = PSOLocalBest(func=rosenbrock, n_dim=n_dim, pop=pop, max_iter=max_iter, lb=lb, ub=ub, verbose=True, X=X, V=V)
lpso.record_mode = True
lpso.run()
```

## Limitation
Currently supports dimensions up to 3. This means that the number of variables in the optimization problem cannot exceed 3. Support for higher dimensions may be added in future updates.

## References
- [1] Z. Zhan, J. Zhang, Y. Li, and Y. Shi, Orthogonal learning particle swarm optimization, IEEE Transactions on Evolutionary Computation, vol. 15, no. 6, pp. 832-847, 2011.
- [2] PSO code: [scikit-opt](https://github.com/guofei9987/scikit-opt)

## Contributing
Contributions are always welcome! Please feel free to open issues or pull requests.