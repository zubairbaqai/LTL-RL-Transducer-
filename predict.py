import gym
import tensorflow as tf
import baselines.common.tf_util as U
from baselines import deepq


def main():
    print('-*-*-*- enjoy worker -*-*-*-')
    # tf.graph().as_default()
    # tf.reset_default_graph()
    env = gym.make("CartPole-v0")
    act = deepq.load_act("model.pkl")

    max_episodes = 5

    while max_episodes > 0:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)
        max_episodes = max_episodes - 1


if __name__ == '__main__':
    main()