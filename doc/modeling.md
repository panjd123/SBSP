# Modeling

| Symbol | Description |
| --- | --- |
| $r_i$ | arrival time |
| $l_i$ | length of the ship |
| $p_i$ | working time |
| $d_i$ | draft of the ship |
| $b_j$ | length of the berth |
| $a$ | amplitude of the tide |
| $T$ | period of the tide |
| $D^t_j$ | depth of the water at time $t$ |
| | $D^t_j = D^0_j + a \sin(\frac{2\pi t}{T})$ |
| $N$ | number of ships |
| $M$ | number of berths |
| $V$ | $\{1, 2, \dots, N\}$ |
| $B$ | $\{1, 2, \dots, M\}$ |
| $O$ | $\{1, 2, \dots, N\}$ |
| $P_i$ | $\{0, 1, \dots, p_i-1\}$ |

## Objective function

Version 1

| Decision variables | Description |
| --- | --- |
| $x_{ijk}$ | 1 if ship $i$ is the $k$-th ship to be assigned to berth $j$, 0 otherwise |
| $t_{i}$ | start working time of ship $i$ |


$$
\min \sum_{i \in V}(t_i - r_i) \\
\text{s.t.} \\
\begin {align} 
\sum_{j \in B, k \in O} x_{ijk} = 1, \forall i \in V \\
\sum_{i \in V} x_{ijk} \leq 1, \forall j \in B, k \in O \\
r_i - t_i \leq 0, \forall i \in V \\
\sum_{i \in V} x_{ijk} - \sum_{i \in V} x_{ijk+1} \geq 0, \forall j \in B, k,k+1 \in O \\
x_{ijk}x_{ij'k+1}(t_i + p_i - t_{i'}) \leq 0, \forall i, i' \in V, j \in B, k,k+1 \in O \\
x_{ijk}(l_i-b_j) \leq 0, \forall i \in V, j \in B, k \in O \\
x_{ijk}(d_i - D_j^{t_i+u}) \leq 0, \forall i \in V, j \in B, k \in O, u \in P_i \\
x_{ijk} \in \{0, 1\}, \forall i \in V, j \in B, k \in O \\
t_i \geq 0, \forall i \in V
\end {align}
$$

(1) Each ship is assigned to exactly one berth.

(2) Each berth can only be assigned to one ship at a time.

(3) The start working time of each ship is no earlier than its arrival time.

(4) True if 1 1 or 1 0 or 0 0, that is making sure k is continuous.

(5) The working time of two ships assigned to the same berth cannot overlap.

(6) The draft of the ship cannot exceed the depth of the water at the time it starts working.

(7) The decision variables are binary.

(8) The start working time of each ship is non-negative.

----

Version 2

| Decision variables | Description |
| --- | --- |
| $x_{ij}$ | 1 if ship $i$ is assigned to berth $j$, 0 otherwise |
| $t_{i}$ | start working time of ship $i$ |
| $y_{ii'}$ | middle variable |


$$
\min \sum_{i \in V}(t_i - r_i) \\
\text{s.t.} \\
\begin {align} 
\sum_{j \in B} x_{ij} = 1, \forall i \in V \\
r_i - t_i \leq 0, \forall i \in V \\
x_{ij}x_{i'j}(t_i' + p_i' - t_{i}) \leq My_{ii'}, \forall i, i' \in V, j \in B \\
x_{ij}x_{i'j}(t_i + p_i - t_{i'}) \leq M(1-y_{ii'}), \forall i, i' \in V, j \in B \\
x_{ij}(l_i-b_j) \leq 0, \forall i \in V, j \in B \\
x_{ij}(d_i - D_j^{t_i+u}) \leq 0, \forall i \in V, j \in B, u \in P_i \\
x_{ij} \in \{0, 1\}, \forall i \in V, j \in B\\
y_{ii'} \in \{0, 1\}, \forall i, i' \in V \\
t_i \geq 0, \forall i \in V
\end {align}
$$
