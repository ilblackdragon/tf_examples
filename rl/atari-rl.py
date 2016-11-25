import random
import time

import numpy as np
import skimage.transform
import skimage.color
import gym
import tensorflow as tf


def run_episode(env, agent, max_steps, shape=(64, 64), render_every=None):
  observation = env.reset()
  agent.reset(observation)
  reward = 0.0
  done = False
  for step in range(max_steps):
    action = agent.act(observation, reward)
    observation, reward, done, _ = env.step(action)
    if done:
      break
    if render_every is not None and step % render_every == 0:
      env.render()


class RandomAgent(object):
  
  def __init__(self, action_space):
    self.action_space = action_space

  def act(self, observation, reward):
    return self.action_space.sample()


def sample_final_epsilon():
    """
    Sample a final epsilon value to anneal towards from a distribution.
    These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
    """
    final_epsilons = np.array([.1,.01,.5])
    probabilities = np.array([0.4,0.3,0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]


def process_observation(observation, shape):
  return skimage.transform.resize(skimage.color.rgb2gray(observation), shape)


class TFAgent(object):

  def __init__(self, model_fn, action_space, trace_length, shape=(64, 64)):
    self.action_space = action_space
    self.model_fn = model_fn
    self.trace_length = trace_length
    self.shape = shape
    self.epsilon = 1.0
    self.final_epsilon = sample_final_epsilon() 
    self.graph = self._create_graph()
    self.episode = 0
    self.episode_reward = 0.0
    with self.graph.as_default():
      self.session = tf.contrib.learn.monitored_session.MonitoredSession()

  def _create_graph(self):
    graph = tf.Graph()
    with graph.as_default():
      params = {'n_actions': self.action_space.n}
      self.features = tf.placeholder(
        shape=[None, self.trace_length] + list(self.shape), dtype=tf.float32, name='observation')
      self.targets = {
        'reward': tf.placeholder(shape=[None], dtype=tf.float32, name='reward'),
        'action': tf.placeholder(shape=[None], dtype=tf.int64, name='action')}
      self.prediction, self.loss, self.train_op = self.model_fn(
        self.features, self.targets, 'train', params)
    return graph

  def reset(self, observation):
    print("Episode %d, Reward: %.2f, Epsilon: %.4f" % (self.episode, self.episode_reward, self.epsilon))
    observation = process_observation(observation, self.shape)
    self.observation_trace = [observation] * self.trace_length
    self.last_action = None
    self.episode_reward = 0.0
    self.final_epsilon = sample_final_epsilon()
    self.episode += 1

  def act(self, observation, reward):
    observation = process_observation(observation, self.shape)
    if self.last_action is not None:
      _, loss = self.session.run([self.train_op, self.loss], {
        self.features: [self.observation_trace],
        self.targets['action']: [self.last_action],
        self.targets['reward']: [reward]})
    self.observation_trace = self.observation_trace[1:] + [observation]
    if random.random() <= self.epsilon:
      action = self.action_space.sample()
    else:
      action = self.session.run(self.prediction, {
        self.features: [self.observation_trace]})[0]
    if self.epsilon > self.final_epsilon:
      self.epsilon -= (1.0 - self.final_epsilon) / 10000
    self.last_action = action
    self.episode_reward += reward
    return action
 

def simple_model(features, targets, mode, params):
  n_actions = params.pop('n_actions')

  # DQN model.
  features = tf.contrib.layers.convolution2d(features, 16, 
    kernel_size=[8, 8], stride=[4, 4], padding='SAME',
    activation_fn=tf.nn.relu)
  features = tf.contrib.layers.convolution2d(features, 8, 
    kernel_size=[4, 4], stride=[2, 2], padding='SAME',
    activation_fn=tf.nn.relu)
  features = tf.contrib.layers.flatten(features)
  features = tf.contrib.layers.fully_connected(
    features, 256, activation_fn=tf.nn.relu)
  q_values = tf.contrib.layers.fully_connected(
    features, n_actions, activation_fn=None)
  prediction = tf.argmax(q_values, dimension=1)

  # Compute loss and add optimizer.
  reward, action = targets['reward'], targets['action']
  action = tf.one_hot(action, n_actions, 1.0, 0.0)
  action_q_values = tf.reduce_sum(
    tf.mul(q_values, action), reduction_indices=[1])
  loss = tf.contrib.losses.mean_squared_error(action_q_values, reward) 
  train_op = tf.contrib.layers.optimize_loss(
    loss, tf.contrib.framework.get_global_step(),
    learning_rate=0.01, optimizer='Adam')
  return prediction, loss, train_op


def main():
  env = gym.make('Breakout-v0')
#  agent = RandomAgent(env.action_space)
  agent = TFAgent(simple_model, env.action_space, 10, (64, 64))
  for i in range(100):
    run_episode(env, agent, 100, render_every=10)
    print("episode", i)


if __name__ == "__main__":
  main()

