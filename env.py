''' Defines an environment for TAMER task
MDP:
S: pos X velocity
A: acceleration +-1, 0
T: pos + velocity
R: xxxxxxxxxxxxx[+1]xx[-1]
gamma: 1 (episodic)
'''
import numpy as np
import copy
import random
import math
import agent
import sys, time
from IPython import embed

import pygame
from pygame.locals import *

class Env:
    def __init__(self, size, max_vel_allowed):
    	'''a 1D grid world'''
    	self.size = size
    	self.ball_pos = self.size - 1
    	self.good_pos = self.ball_pos - 3
    	self.max_vel = max_vel_allowed

    def feedback(self, s, a):
    	'''input: s = [pos, vel], a; output: s', r, terminate
    	reward is +10 when at good_pos with velocity 0; -10 when step into ball regardless of velocity; -1 ow'''
    	pos, vel = s[0], s[1]
    	vel_ = vel + a
    	# cannot exceed +- max_vel
    	vel_ = min(max(vel_, -self.max_vel), self.max_vel)
    	pos_ = pos + vel_
    	# handle boundaries
    	pos_ = min(max(pos_, 0), self.size - 1)
    	s_ = [pos_, vel_]
    	# determine reward
    	if pos_ == self.ball_pos:
    		return -10, s_, True
    	elif pos_ == self.good_pos and vel_ == 0:
    		return 10, s_, True
    	else: return -1, s_, False

    def draw_init(self, agent):
    	pygame.init()
    	self.unit = 30
    	pygame.font.init()
    	self.myfont = pygame.font.SysFont('Comic Sans MS', self.unit)
    	self.x, self.y = self.size * self.unit, self.size * self.unit / 2
    	self.screen = pygame.display.set_mode((self.x, self.y))
    	self.clock = pygame.time.Clock()

    def draw_update(self, agent):
    	self.screen.fill((0, 0, 0))
    	# draw rects for each grid
    	for i in range(self.size):
    		x = i * self.unit
    		y = self.unit
    		# label coordinates
    		textsurface = self.myfont.render(str(i), False, (0, 128, 255))
    		self.screen.blit(textsurface, (x + self.unit / 4, 0))
    		if i == self.ball_pos: 
    			color, width = (255,0,0), 0
    		elif i == self.good_pos: 
    			color, width = (0, 255, 0), 0
    		else: 
    			color, width = (0, 128, 255), 1
    		pygame.draw.rect(self.screen, color, pygame.Rect(x, y, self.unit, self.unit), width)
    	# draw agent
        pygame.draw.circle(self.screen, (0, 255, 255), (int(agent.get_pos() * self.unit + self.unit / 2), int(self.unit * 1.5)), self.unit / 4)
    	
    	# display velocity information
    	textsurface = self.myfont.render("Velocity: ", False, (0, 255, 255))
    	self.screen.blit(textsurface, (0, self.unit * 2))
    	for i in range(self.max_vel * 2 + 1):
    		x = i * self.unit
    		textsurface = self.myfont.render(str(i - self.max_vel), False, (0, 255, 255))
    		self.screen.blit(textsurface, (x + self.unit / 4, self.unit * 3))
    	
    	v = agent.get_vel()
    	print("Current v: ", v)
    	start_fill = v + self.max_vel if v < 0 else self.max_vel
    	end_fill = start_fill + abs(v) - 1
    	
    	for i in range(self.max_vel * 2):
    		x = i * self.unit
    		width = 0 if start_fill <= i <= end_fill else 1
   		

    		pygame.draw.rect(self.screen, (0, 255, 255), pygame.Rect(x + self.unit / 2, self.unit * 4, self.unit, self.unit), width)

    	self.clock.tick(5)
    	pygame.display.flip()

if __name__ == '__main__':
	SIZE = 20
	MAX_VEL = 5 # +-
	ACTIONS = [-1,0,1]
	EPSILON = 0.01
	EPISODES = 200
	ALPHA = 0.1 # q learning learning rate
	GAMMA = 0.99


	env = Env(SIZE, MAX_VEL)
	agent = agent.RL_Agent(SIZE, MAX_VEL, ACTIONS, EPSILON, ALPHA, GAMMA)


	for eps in range(EPISODES):
		DRAW = True #(eps == EPISODES - 1)
		terminated = False
		agent.reset_pos()
		cum_reward = 0
		if DRAW: env.draw_init(agent)

		while not terminated:
			a = agent.select_act()
			s = agent.get_state()
			r, s_, terminated = env.feedback(s, a)
			agent.update_q(a, r, s_)
			agent.update_state(s_)
			cum_reward += r
			if DRAW: env.draw_update(agent)
			
		print("Episode %d ends with total reward %d" %(eps, cum_reward))

