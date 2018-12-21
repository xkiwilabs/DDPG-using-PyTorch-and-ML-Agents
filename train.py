
"""
DDPG (Actor-Critic) RL Example for Unity ML-Agents Environments using PyTorch
Includes examples of the following DDPG training algorithms:

The example uses a modified version of the Unity ML-Agents Reacher Example Environment.
The environment includes In this environment, a double-jointed arm can move to target locations. 
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 
Thus, the goal of your agent is to maintain its position at the target location for as many 
time steps as possible.

Example Developed By:
Michael Richardson, 2018
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code Expanded and Adapted from Code provided by Udacity DRL Team, 2018.
"""

###################################
# Import Required Packages
import torch
import random
import numpy as np
from collections import deque
from ddpg_agent import Agent
from unityagents import UnityEnvironment

"""
###################################
STEP 1: Set the Training Parameters
======
        num_episodes (int): maximum number of training episodes
        episode_scores (float): list to record the scores obtained from each episode
        scores_average_window (int): the window size employed for calculating the average score (e.g. 100)
        solved_score (float): the average score required for the environment to be considered solved
    """
num_episodes=500
episode_scores = []
scores_average_window = 100      
solved_score = 30     

"""
###################################
STEP 2: Start the Unity Environment
# Use the corresponding call depending on your operating system 
"""
env = UnityEnvironment(file_name="Reacher.app")
# - **Mac**: "Banana_Mac/Reacher.app"
# - **Windows** (x86): "Reacher_Windows_x86/Reacher.exe"
# - **Windows** (x86_64): "Reacher_Windows_x86_64/Reacher.exe"
# - **Linux** (x86): "Reacher_Linux/Reacher.x86"
# - **Linux** (x86_64): "Reacher_Linux/Reacher.x86_64"
# - **Linux** (x86, headless): "Reacher_Linux_NoVis/Reacher.x86"
# - **Linux** (x86_64, headless): "Reacher_Linux_NoVis/Reacher.x86_64"

"""
#######################################
STEP 3: Get The Unity Environment Brian
Unity ML-Agent applications or Environments contain "BRAINS" which are responsible for deciding 
the actions an agent or set of agents should take given a current set of environment (state) 
observations. The Reacher environment has a single Brian, thus, we just need to access the first brain 
available (i.e., the default brain). We then set the default brain as the brain that will be controlled.
"""
# Get the default brain 
brain_name = env.brain_names[0]

# Assign the default brain as the brain to be controlled
brain = env.brains[brain_name]


"""
#############################################
STEP 4: Determine the size of the Action and State Spaces and the Number of Agents

The observation space consists of 33 variables corresponding to
position, rotation, velocity, and angular velocities of the arm. 
Each action is a vector with four numbers, corresponding to torque 
applicable to two joints. Every entry in the action vector should 
be a number between -1 and 1.

The reacher environment can contain multiple agents in the environment to increase training time. 
To use multiple (active) training agents we need to know how many there are.
"""

# Set the number of actions or action size
action_size = brain.vector_action_space_size

# Set the size of state observations or state size
state_size = brain.vector_observation_space_size

# Get number of agents in Environment
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
print('\nNumber of Agents: ', num_agents)


"""
###################################
STEP 5: Create a DDPG Agent from the Agent Class in ddpg_agent.py
A DDPG agent initialized with the following parameters.
    ======
    state_size (int): dimension of each state (required)
    action_size (int): dimension of each action (required)
    num_agents (int): number of agents in the unity environment
    seed (int): random seed for initializing training point (default = 0)

Here we initialize an agent using the Unity environments state and action size and number of Agents
determined above.
"""
agent = Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, random_seed=0)


"""
###################################
STEP 6: Run the DDPG Training Sequence
The DDPG Training Process involves the agent learning from repeated episodes of behaviour 
to map states to actions the maximize rewards received via environmental interaction.

The agent training process involves the following:
(1) Reset the environment at the beginning of each episode.
(2) Obtain (observe) current state, s, of the environment at time t
(3) Perform an action, a(t), in the environment given s(t)
(4) Observe the result of the action in terms of the reward received and 
	the state of the environment at time t+1 (i.e., s(t+1))
(5) Update agent memory and learn from experience (i.e, agent.step)
(6) Update episode score (total reward received) and set s(t) -> s(t+1).
(7) If episode is done, break and repeat from (1), otherwise repeat from (3).

Below we also exit the training process early if the environment is solved. 
That is, if the average score for the previous 100 episodes is greater than solved_score.
"""

# loop from num_episodes
for i_episode in range(1, num_episodes+1):

    # reset the unity environment at the beginning of each episode
    env_info = env.reset(train_mode=True)[brain_name]     

    # get initial state of the unity environment 
    states = env_info.vector_observations
    
	# reset the training agent for new episode
    agent.reset()

    # set the initial episode score to zero.
    agent_scores = np.zeros(num_agents)

    # Run the episode training loop;
    # At each loop step take an action as a function of the current state observations
    # Based on the resultant environmental state (next_state) and reward received update the Agents Actor and Critic networks
    # If environment episode is done, exit loop...
    # Otherwise repeat until done == true 
    while True:
        # determine actions for the unity agents from current sate
        actions = agent.act(states)

        # send the actions to the unity agents in the environment and receive resultant environment information
        env_info = env.step(actions)[brain_name]        

        next_states = env_info.vector_observations   # get the next states for each unity agent in the environment
        rewards = env_info.rewards                   # get the rewards for each unity agent in the environment
        dones = env_info.local_done                  # see if episode has finished for each unity agent in the environment

        #Send (S, A, R, S') info to the training agent for replay buffer (memory) and network updates
        agent.step(states, actions, rewards, next_states, dones)

        # set new states to current states for determining next actions
        states = next_states

        # Update episode score for each unity agent
        agent_scores += rewards

        # If any unity agent indicates that the episode is done, 
        # then exit episode loop, to begin new episode
        if np.any(dones):
            break

    # Add episode score to Scores and...
    # Calculate mean score over last 100 episodes 
    # Mean score is calculated over current episodes until i_episode > 100
    episode_scores.append(np.mean(agent_scores))
    average_score = np.mean(episode_scores[i_episode-min(i_episode,scores_average_window):i_episode+1])

    #Print current and average score
    print('\nEpisode {}\tEpisode Score: {:.3f}\tAverage Score: {:.3f}'.format(i_episode, episode_scores[i_episode-1], average_score), end="")
    
    # Save trained  Actor and Critic network weights after each episode
    an_filename = "ddpgActor_Model.pth"
    torch.save(agent.actor_local.state_dict(), an_filename)
    cn_filename = "ddpgCritic_Model.pth"
    torch.save(agent.critic_local.state_dict(), cn_filename)

    # Check to see if the task is solved (i.e,. avearge_score > solved_score over 100 episodes). 
    # If yes, save the network weights and scores and end training.
    if i_episode > 100 and average_score >= solved_score:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(i_episode, average_score))

        # Save the recorded Scores data
        scores_filename = "ddpgAgent_Scores.csv"
        np.savetxt(scores_filename, episode_scores, delimiter=",")
        break


"""
###################################
STEP 7: Everything is Finished -> Close the Environment.
"""
env.close()

# END :) #############

