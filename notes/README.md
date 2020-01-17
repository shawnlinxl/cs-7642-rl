# Reinforcement Learning

## Decision Making and Reinforcement Learning

- Supervised Learning: $y = f(x)$
  - function approximation: given $x$, $y$ pairs, find function $f$ that maps $x$ to $y$
- Unsupervised Learning: $f(x)$
  - find $f$ that is a compact description of data x
- Reinforcement Learning: $y = f(x), z$
  - Given a string of pairs of data
  - Given $x$ and $z$'s, learn $f$ that generates $y$

## Markov Decision Processes

[Wikipedia](https://en.wikipedia.org/wiki/Markov_decision_process) Definition:
![MDP Definition](img/1.6_MDP_Definition.png)

### The problem

- State: $S$
- Actions: $A(s), A$
- Model
  - Rules of the game you are playing. Transition model $T$ describes how to transition from state $s^\prime$ to state $s$ given actions $a$.
  - "The physics of the world"
  - Stationary: rules don't change, so the transition models don't change
  - $T(s,a,s^\prime) ~ Pr(s^\prime|s,a)$
- Reward: $R(s), R(s,a), R(s,a,s^\prime)$
  - A scalar value for being in the state
  - $R(s)$: reward for being in a state
  - $R(s, a)$: reward for being in a state and taking an action
  - $R(s, a, s^\prime)$ reward for being in a state, taking an action, and end up in another state

Markov property:

- only the present matters
- transitions only depends the current state $s$
- A stochastic process has the Markov property if the conditional probability distribution of future states of the process (conditional on both past and present states) depends only upon the present state, not on the sequence of events that preceded it. A process with this property is called a Markov process.

### The solution

- Policy:
  - For any given state, it gives the action you should take
  - $\pi(s) \rightarrow a$
  - $\pi^*$ is the optimal policy which maximizes long term expected reward