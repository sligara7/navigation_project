#### Udacity Deep Reinforcement Learning Nanodegree
### Project 1: Navigation
# Train an RL Agent to Navigate and Collect Yellow Bananas

##### &nbsp;

## Goal
In this project, I build a reinforcement learning (RL) agent that navigates an environment that is similar to [Unity's Banana Collector environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector).

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas. In order to solve the environment, our agent must achieve an average score of +13 over 100 consecutive episodes.

##### &nbsp;

## Approach
Below are a list of the general steps taken to train an agent:

1. Examine the environment, states, and actions.
2. Take some random actions within the environment.
3. Finally, set up the Deep Q-Network to train the agent.

##### &nbsp;

### 1. Evaluate State & Action Space
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available:

- `0` move forward
- `1` move backward
- `2` turn left
- `3` turn right


##### &nbsp;

### 2. Run environment
Before building an agent that learns, test an agent that selects actions (uniformly) at random at each time step.

```python
env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = np.random.randint(action_size)        # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break

print("Score: {}".format(score))
```
Running this code shows that learning is required and that it cannot be solved using a series of random (uniform) actions at each step in an episode.  


##### &nbsp;

### 3. Implement Learning Algorithm
The goal of the agent is to maximize reward in a given episode.  In general, an environment contains a set of states.  The agent will take an action, `A`, at each state, `S`  and earn a reward, `R`.  Often the states and actions are paired together in a state, action pair within a tuple. The goal of the agent over a series of finite episodes is to learn which actions maximize cummulative reward over an episode.  The knowledge of which action to take in each state, is a policy, `π`.  The policy that maximizes the total expected reward is called the optimal policy, `π*`.  As the optimal policy is not known in advance, the agent must interact and learn this through a series of trial and error. This type of algorithm is called **Q-Learning**.  The optimal Q-function `Q*(s,a)` maximizes the total expected reward for an agent starting in state `s` and choosing action `a`.

There are various reinforcement learning techniques to obtain the optimal policy - this project utilizes Temporal Differencing, which updates the policy with each step within an episode, instead of waiting to learn until an episode is completed.  As the this is a stochastic environment and it cannot be known the exact return, a discount factor is applied to the expected return of each future step - the discount factor is denoted by the hyperparameter gamma `γ`.

#### Epsilon Greedy Algorithm
Reinforcement Learning is affected by the **exploration vs. exploitation dilemma**.  In a totally greedy control, the model would learn to simply select the action that has the maximum expected return `argmax a` without ever exploring other possibliities.  Particularly, when learning has just started, the optimal policy has not been determined and a suboptimal action will continue to be selected again and again.  To ensure that other actions are taken to explore other possible actions, an **𝛆-greedy algorithm** is utilized.  The agent "explores" by picking a random action with some probability epsilon `𝛜`; else, it takes the **greedy action** with a probability of (1-𝛜).  As the model learns or converges towards the optimal policy, `𝛜` is decayed - in effect, it takes the **greedy action** with a greater and greater probability.  Within the Agent class, 𝛆-greedy logic implemented as part of the `agent.act()` method.


#### Deep Q-Network (DQN)
With Deep Q-Learning, a deep neural network,`NN`, is used to approximate the Q-function. DQN utilizes a NN to approximate a function,`F`.  The optimal policy is the function where `F(s,a,w) ≈ Q(s,a)`. The weights, `w`, become the parameter that the NN minimizes error.  A DQN is designed to produce a Q-value for every possible action in a single forward pass.   

#### Experience Replay
The first techIn a simple DQN, the interaction is learned from and then discarded, which is wasteful.  The idea of experience replay is to store a sampling of `(s,a, r, s')` in a buffer to relearn from them.  This also helps prevent an issue that occurs when a sequence of states and actions become highly correlated, which can cause a level of oscillation or divergence in a DQN.  The experience replay is setup in the class called `ReplayBuffer` within the Navigation.ipynb code.


##### &nbsp;

## Future Improvements
There are many techniques in addition to experience replay designed to deal with a other issues that arize in training a DQN.  These include utilizing a Double DQN, Prioritized Experience Replay, Duelling DQN, multistep bootstrap targets, distributional DQN, and noisy DQN.  Additionally, all these techniques can be combined together - one example of this is the rainbow DQN, which can be found at https://github.com/Kaixhin/Rainbow.  If a cost to benefit ratio can be considered where the amount of work put into making these improvements, the benefit may be marginal.  A simple DQN utilizing experience replay seems to be sufficient for this task.  

##### &nbsp;
##### &nbsp;
