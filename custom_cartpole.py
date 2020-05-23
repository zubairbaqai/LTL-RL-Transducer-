import gym
import itertools
import numpy as np
import tensorflow as tf


from gym_breakout_pygame.wrappers.normal_space import BreakoutNMultiDiscrete
from gym_breakout_pygame.breakout_env import BreakoutConfiguration
from gym_breakout_pygame.wrappers.dict_space import BreakoutDictSpace
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule
from stable_baselines.deepq.policies import MlpPolicy




from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)

        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out



if __name__ == '__main__':
    with U.make_session(num_cpu=8):
        # Create the environment
        #env = gym.make("Breakout-v0")
        #env=BreakoutNMultiDiscrete()
        env = gym.make("CartPole-v0")
        print(env.action_space.sample())
        print(env.observation_space.high)
        print(env.observation_space.low)
        print(env.action_space.n)


        





    #     env = BreakoutDictSpace()
        




    #     config = BreakoutConfiguration(
    #     brick_rows=3,
    #     brick_cols=3,
    #     fire_enabled=True,
    #     ball_enabled=True,
    # )
    #     env = BreakoutDictSpace(config)






        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
            q_func=model,
            num_actions=env.action_space.n,
            # gamma=0.98,
            # grad_norm_clipping=10,
            optimizer=tf.train.AdamOptimizer(learning_rate=1e-3),
        )


        print("NOP")
        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=100000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        obs = env.reset()
        episodeCounter=0
        for t in itertools.count():
            
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(t))[0]
            #print(action)
            new_obs, rew, done, _ = env.step(action)
            

            #new_obs=new_obs

            #print(new_obs)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew


            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if episodeCounter % 100 == 0 or episodeCounter<1:
                # Show off the result
                #print("coming here Again and Again")
                env.render()


            if done:
                episodeCounter+=1
                obs = env.reset()
                episode_rewards.append(0)
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))

                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()

            if done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

