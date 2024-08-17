# spring

Implementation of "Graph Drawing by Force-directed Placement" by Frutcherman and Reingold
using the jit-jax. This repo interprets the procedural/step-by-step nature of the algorithm
into a matrix and gpu-friendly manner. 

Most of the procedures and formulae have been merged into the equation below.

For a Graph $G = <V, E>$, let $\Delta = V[None,:,:] - V[:, None,:]$ pairwise vectors from $i$ to $j$ for any $\Delta_{ij}$.

The attractive displacement is $F_A = \frac{E \Delta |\Delta|}{k}$ and the repulsive displacement is $F_R = -\frac{\Delta k^2}{ |\Delta|^2}$. And the total displament for each pairwise interactions is $D$.

$$D = \frac{E\Delta|\Delta|^3 - \Delta k^3}{k |\Delta|^2}$$

At this point $D \in \mathbb{R}^{n \times n \times 2}$ and shows the displacement contribution of each pair of nodes. $d = \Sum_j D_j$ where $d \in \mathbb{R}^{n \times 2}$ is the raw update vector for V. 

The last part of the algorithm is to cap the updates in either the $x$ or $y$ direction by the temperature $\tau$ at time $t$.

