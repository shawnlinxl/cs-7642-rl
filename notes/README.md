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

### More about rewards

- Delayed reward
- Minor changes in the reward function matter

[Temporal credit assignment](https://ai.stackexchange.com/questions/12908/what-is-the-credit-assignment-problem):

> In reinforcement learning (RL), an agent interacts with an environment in time steps. On each time step, the agent takes an action in a certain state and the environment emits a percept or perception, which is composed of a reward and an observation, which, in the case of fully-observable MDPs, is the next state (of the environment and the agent). The goal of the agent is to maximize the reward in the long run.
>
> The (temporal) credit assignment problem (CAP) (discussed in Steps Toward Artificial Intelligence by Marvin Minsky in 1961) is the problem of determining the actions that lead to a certain outcome.
>
> For example, in football, at each second, each football player takes an action. In this context, an action can e.g. be "pass the ball", "dribble", "run" or "shoot the ball". At the end of the football match, the outcome can either be a victory, a loss or a tie. After the match, the coach talks to the players and analyses the match and the performance of each player. He discusses the contribution of each player to the result of the match. The problem of determining the contribution of each player to the result of the match is the (temporal) credit assignment problem.
>
> How is this related to RL? In order to maximize the reward in the long run, the agent needs to determine which actions will lead to such outcome, which is essentially the temporal CAP.
>
> Why is it called credit assignment problem? In this context, the word credit is a synonym for value. In RL, an action that leads to a higher final cumulative reward should have more value (so more "credit" should be assigned to it) than an action that leads to a lower final reward.
>
> Why is the CAP relevant to RL? Most RL agents attempt to solve the CAP. For example, a Q-learning agent attempts to learn an (optimal) value function. To do so, it needs to determine the actions that will lead to the highest value in each state.
>
> There are a few variations of the (temporal) CAP problem. For example, the structural CAP, that is, the problem of assigning credit to each structural component (which might contribute to the final outcome) of the system.

Reward functions can affect the MDP
![quiz](img/1.12_Different_Reward_Functions_and_Paths.png)

## Sequence of Rewards

Previous assumptions:

- Infinite horizons: game doesn't end until reaching an absorbent state
- Utility of sequences:

  $$
  $$

  $$U(S_0,S1,S2 ...) = \sum_{t = 0}^\infin \gamma^t R(S_t) \leq \sum_{t = 0}^\infin \gamma^t R_{\text{max}} = \frac{R_{\text{max}}}{1 - \gamma} \text{ where } 0 \leq \gamma < 1$$

## Policies

$$\pi^* = \arg \max_\pi E[\sum_{t = 0}^\infin \gamma^t R(S_t) | \pi]$$
$$U^\pi(s) =E[\sum_{t = 0}^\infin \gamma^t R(S_t) | \pi, S_0 = S] $$

Note: $R(S)$ is an immediate reward, while $U^\pi(S)$ is the long term expected benefit. $U^pi(S)$ accounts for all delayed awards.

$$\pi^*(S) = \arg \max_a \sum_{S^\prime} T(S,a,S^\prime) U(S^\prime)$$

### Finding Policies

$$\text{Bellman Equation: }U(S) = R(S) + \gamma \max_a \sum_{S^\prime} T(S,a,S^\prime) U(S^\prime)$$

- $n$ equations
- $n$ unknowns
- $\max$ introduces non-linearity

Solving the system of equations:

#### Value iteration

- start with arbitrary utilities
- update utilities based on neighbors
- repeat until convergence
  i.e. \\$\$\hat{U}\_{t+ 1}(S) = R(S) + \gamma \max_a \sum\_{S^\prime} T(S,a,S^\prime) \hat{U}\_t(S^\prime)\\\text{ where } \hat{U}\_t(S) \text{ is the estimate of the utility of \$S\$ at time \$t\$.}\$\$
- Intuition: $R(S)$ is a truth, and in every iteration truth is added to estimates. As more truth is added, you move closer to the true utility of states.
  ![Finding Policies](img/1.22_Finding_Policies_Value_Iter_Quiz.png)

#### Policy Iteration

- start with $\pi_0$, a random guess
- evaluate: given $\pi_t$, calculate $U_t = U^{\pi_t}$
- improve: $\pi_{t+1} = \arg\max_a \sum T(S,a,S^\prime)U_t(S^\prime)$
- Now a linear equation: $\hat{U}_{t+ 1}(S) = R(S) + \gamma \sum_{S^\prime} T(S,\pi_t(s),S^\prime) \hat{U}_t(S^\prime)$

## The Bellman Equation

![Bellman Equation](img/1.25_Bellman_Equation.png)

And this is "Value" version of the Bellman equation is equivalent to the "Quality" version of the equation
$$ Q(S,a) = R(S,a) + \gamma\sum_{S^\prime} T(S,a,S^\prime)\max_{a^\prime}Q(S^\prime, a^\prime)$$
or the "Continuation" version of the equation
$$C(S,a) = \gamma \sum_{S^\prime} T(S,a,S^\prime)\max_{a^\prime}(R(S^\prime, a^\prime) + C(S^\prime, a^\prime)).$$
