---
orphan: true
---

# Parity

Use parity when you have two simplices and need vertices at which to join one to
the other. Two ordered vertex sets $\mathcal{V}_i$ and $\mathcal{V}_j$ *have
parity* when there are ordered subsets $V_i \subseteq \mathcal{V}_i$ and
$V_j \subseteq \mathcal{V}_j$ such that `Vertex::getTime` returns the same value
for every pair of corresponding elements. Formally, $V_i$ and $V_j$ have parity
if and only if

- they have the same cardinality $N$, and
- for each index $m \in \{1, \dots, N\}$, the $m$-th element $v_m \in V_i$ and
  the $m$-th element $w_m \in V_j$ satisfy
  $v_m.\mathrm{getTime}() = w_m.\mathrm{getTime}()$.
