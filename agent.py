import numpy as np
import copy
import random
import math

class Agent(object):
	'''A simple learning agent'''
	def __init__(self, env_size, max_vel_allowed, actions):
		self.pos = 0
		self.vel = 0
		self.env_size = env_size
		self.max_vel = max_vel_allowed
		self.actions = actions

	def get_pos(self):
		return self.pos

	def get_vel(self):
		return self.vel

	def reset_pos(self):
		self.pos, self.vel = 0, 0

	def get_state(self):
		return [self.pos, self.vel]

	def update_state(self, s):
		self.pos, self.vel = s[0], s[1]
	
	def s_index(self, s):
		'''map pos and vel to q table index; note: return a tuple'''
		return [s[0], s[1] + self.max_vel]

	def a_index(self, a):
		'''map action to q table index'''
		return a + len(self.actions) / 2

class RL_Agent(Agent):
	def __init__(self, env_size, max_vel_allowed, actions, epsilon, alpha, gamma):
		super(RL_Agent, self).__init__(env_size, max_vel_allowed, actions)
		self.q = np.zeros((self.env_size, self.max_vel * 2 + 1, len(self.actions)))
		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma

	def select_act(self):
		'''e-greedy choice of action according to current state [pos, vel], note this returns actual action, not its index'''
		# get value vector for the current state from q table
		s_index_x, s_index_y = self.s_index([self.pos, self.vel])
		q_s = self.q[s_index_x, s_index_y]
		if random.random() < self.epsilon:
			return self.actions[random.randint(0, len(self.actions) - 1)]
		else:
			return self.actions[np.argmax(q_s)]

	def update_q(self, a, r, s_):
		s = [self.pos, self.vel]
		s_index_x, s_index_y = self.s_index(s)
		q_s_a = self.q[s_index_x, s_index_y, self.a_index(a)]
		s_prime_index_x, s_prime_index_y = self.s_index(s_)
		a_prime_index = np.argmax(self.q[s_prime_index_x, s_prime_index_y])
		q_s_prime_a_prime = self.q[s_prime_index_x, s_prime_index_y, a_prime_index]
		td_err = r + self.gamma * (q_s_prime_a_prime - q_s_a)
		debug = False
		if debug:
			print("s: ", s, " a: ", a, " s_index: ", self.s_index(s), " a_index: ", self.a_index(a))
			print("q(s,a): ", q_s_a)
			print("s': ", s_, "s'_index: ", s_prime_index_x, s_prime_index_y, "a'_index:", a_prime_index)
			print("q(s',a'): ", q_s_prime_a_prime)
			print("r: ", r)
			print("td_err: ", td_err)
			print("Before: ", self.q)
		self.q[s_index_x, s_index_y, self.a_index(a)] += self.alpha * td_err
		if debug:
			print("After", self.q)
			print("===============================")

class Tamer_Agent(RL_Agent):
	def __init__(self, env_size, max_vel_allowed, actions, epsilon, alpha, gamma):
		super(Tamer_Agent, self).__init__(env_size, max_vel_allowed, actions)
		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma
		self.pi = np.zeros((self.env_size, self.max_vel * 2 + 1, len(self.actions)))
		self.history = []

	def update_pi(self, a, r):
		# r is human feedback in Tamer; gamma is the credit assignment over history length
		s = [self.pos, self.vel]
		s_index_x, s_index_y = self.s_index(s)
		pi_s_a = self.pi[s_index_x, s_index_y, self.a_index(a)]
	

if __name__ == '__main__':
	SIZE = 20
	MAX_VEL = 5 # +-
	ACTIONS = [-1,0,1]
	EPSILON = 0.01
	EPISODES = 200
	ALPHA = 0.1 # q learning learning rate
	GAMMA = 0.99

	tamer_agent = Tamer_Agent(SIZE, MAX_VEL, ACTIONS, EPSILON, ALPHA, GAMMA)
	print(tamer_agent.gamma)