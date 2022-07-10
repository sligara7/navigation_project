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
With Deep Q-Learning, a deep neural network is used to approximate the Q-function. Given a network `F`, finding an optimal policy is a matter of finding the best weights `w` such that `F(s,a,w) ≈ Q(s,a)`.

The neural network architecture used for this project can be found [here](https://github.com/tommytracey/DeepRL-P1-Navigation/blob/master/model.py#L5) in the `model.py` file of the source code. The network contains three fully connected layers with 64, 64, and 4 nodes respectively. Testing of bigger networks (more nodes) and deeper networks (more layers) did not produce better results.

As for the network inputs, rather than feeding-in sequential batches of experience tuples, I randomly sample from a history of experiences using an approach called Experience Replay.


#### Experience Replay
Experience replay allows the RL agent to learn from past experience.

Each experience is stored in a replay buffer as the agent interacts with the environment. The replay buffer contains a collection of experience tuples with the state, action, reward, and next state `(s, a, r, s')`. The agent then samples from this buffer as part of the learning step. Experiences are sampled randomly, so that the data is uncorrelated. This prevents action values from oscillating or diverging catastrophically, since a naive Q-learning algorithm could otherwise become biased by correlations between sequential experience tuples.

Also, experience replay improves learning through repetition. By doing multiple passes over the data, our agent has multiple opportunities to learn from a single experience tuple. This is particularly useful for state-action pairs that occur infrequently within the environment.

The implementation of the replay buffer can be found [here](https://github.com/tommytracey/DeepRL-P1-Navigation/blob/master/agent.py#L133) in the `agent.py` file of the source code.


#### Dueling Agents
Dueling networks utilize two streams: one that estimates the state value function `V(s)`, and another that estimates the advantage for each action `A(s,a)`. These two values are then combined to obtain the desired Q-values.  

<img src="assets/dueling-networks-slide.png" width="60%" align="top-left" alt="" title="DDQN" />

The reasoning behind this approach is that state values don't change much across actions, so it makes sense to estimate them directly. However, we still want to measure the impact that individual actions have in each state, hence the need for the advantage function.

The dueling agents are implemented within the fully connected layers [here](https://github.com/tommytracey/DeepRL-P1-Navigation/blob/master/model.py#L21) in the `model.py` file of the source code.


##### &nbsp;

### 4. Run Experiments
Now that the various components of our algorithm are in place, it's time to measure the agent's performance within the Banana environment. Performance is measured by the fewest number of episodes required to solve the environment.

The table below shows the complete set of experiments. These experiments compare different combinations of the components and hyperparameters discussed above. However, note that all agents utilized a replay buffer.

<img src="assets/experiment_summary.png" width="80%" align="top-left" alt="" title="Experiment Summary" />

##### &nbsp;

## Future Improvements
- **Test the replay buffer** &mdash; Implement a way to enable/disable the replay buffer. As mentioned before, all agents utilized the replay buffer. Therefore, the test results don't measure the impact the replay buffer has on performance.
- **Add *prioritized* experience replay** &mdash; Rather than selecting experience tuples randomly, prioritized replay selects experiences based on a priority value that is correlated with the magnitude of error. This can improve learning by increasing the probability that rare and important experience vectors are sampled.
- **Replace conventional exploration heuristics with Noisy DQN** &mdash; This approach is explained [here](https://arxiv.org/abs/1706.10295) in this research paper. The key takeaway is that parametric noise is added to the weights to induce stochasticity to the agent's policy, yielding more efficient exploration.

##### &nbsp;
##### &nbsp;
