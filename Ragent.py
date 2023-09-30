import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class RMaxAgent:
    def __init__(self, env, m=5, gamma=0.99,seed=None):
        np.random.seed(seed)
        self.env = env
        self.m = m  # The threshold for considering a state-action as 'known'
        self.gamma = gamma
        self.q_values = np.full((env.observation_space.n, env.action_space.n), float('inf'))
        self.state_action_counts = np.zeros((env.observation_space.n, env.action_space.n))
        self.total_rewards = np.zeros((env.observation_space.n, env.action_space.n))
        self.correct_choices_phase1 = []
        self.correct_choices_phase3 = []
        
    def select_action(self, state):
        unknown_actions = np.where(self.q_values[state] == float('inf'))[0]
        if len(unknown_actions) > 0:
            return np.random.choice(unknown_actions)
        else:
            return np.argmax(self.q_values[state])
        
    def update_q_values(self):
        # Update Q-values for known state-actions
        for s in range(self.env.observation_space.n):
            for a in range(self.env.action_space.n):
                if self.state_action_counts[s, a] >= self.m:
                    self.q_values[s, a] = self.total_rewards[s, a] / self.state_action_counts[s, a]
    
    def train(self, max_steps=50):
        correct_ratio_phase1_list = []
        correct_ratio_phase3_list = []
        
        phases = ['I', 'II', 'III']
        for phase in phases:
            self.env.set_phase(phase)
            state = self.env.reset()
            correct_choices = 0
            total_choices = 0
            
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, _, _ = self.env.step(action)
                
                # Update state-action count and total rewards
                self.state_action_counts[state, action] += 1
                self.total_rewards[state, action] += reward
                
                # Update Q-values
                self.update_q_values()
                
                # For calculating the correct choices to total choices ratio
                if phase == 'I' or phase == 'III':
                    correct_action = np.argmax([4, 2]) if state == 1 else np.argmax([2, 4])
                    if action == correct_action:
                        correct_choices += 1
                    total_choices += 1

                    if phase == 'I':
                        correct_ratio_phase1_list.append(correct_choices / total_choices)
                    elif phase == 'III':
                        correct_ratio_phase3_list.append(correct_choices / total_choices)
                        
                state = next_state

        # Line plotting the correct ratio for Phase I and Phase III
        plt.plot(range(len(correct_ratio_phase1_list)), correct_ratio_phase1_list, label='Phase I')
        plt.plot(range(len(correct_ratio_phase3_list)), correct_ratio_phase3_list, label='Phase III')
        plt.ylabel('Correct Choices Ratio')
        plt.ylim(0, 1)
        plt.xlabel('Total Choices Made')
        plt.title('Performance of R-max Agent')
        plt.legend()
        plt.show()
        
        return correct_ratio_phase1_list, correct_ratio_phase3_list
