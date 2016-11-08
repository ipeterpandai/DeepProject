# Deep Q network

import gym
import numpy as np
import tensorflow as tf
import math
import random
import bisect

# HYPERPARMETERS
H = 200
H2 = 200
batch_number = 500
gamma = 0.999
num_of_ticks_between_q_copies = 500
explore_decay = 0.9995
min_explore = 0.005
max_steps = 199
max_episodes = 1500
memory_size = 100000
learning_rate = 1e-3

if __name__ == '__main__':
    # Setup open AI gym
    env = gym.make('CartPole-v0')
    env.monitor.start('training_dir', force=True)

    tf.reset_default_graph()

    # First Q Network weights and bias
    w1 = tf.Variable(tf.random_uniform([env.observation_space.shape[0], H], -.10, .10))
    w2 = tf.Variable(tf.random_uniform([H, H2], -.10, .10))
    w3 = tf.Variable(tf.random_uniform([H2, env.action_space.n], -.10, .10))

    b1 = tf.Variable(tf.random_uniform([H], -.10, .10))
    b2 = tf.Variable(tf.random_uniform([H2], -.10, .10))
    b3 = tf.Variable(tf.random_uniform([env.action_space.n], -.10, .10))
    # Second Q Network weights and bias
    w1_prime = tf.Variable(tf.random_uniform([env.observation_space.shape[0], H], -1.0, 1.0))
    w2_prime = tf.Variable(tf.random_uniform([H, H2], -1.0, 1.0))
    w3_prime = tf.Variable(tf.random_uniform([H2, env.action_space.n], -1, 1))

    b1_prime = tf.Variable(tf.random_uniform([H], -1.0, 1.0))
    b2_prime = tf.Variable(tf.random_uniform([H2], -1.0, 1.0))
    b3_prime = tf.Variable(tf.random_uniform([env.action_space.n], -1, 1))

    # Make assign functions for updating Q prime's weights and bias
    w1_prime_update = w1_prime.assign(w1)
    w2_prime_update = w2_prime.assign(w2)
    w3_prime_update = w3_prime.assign(w3)

    b1_prime_update = b1_prime.assign(b1)
    b2_prime_update = b2_prime.assign(b2)
    b3_prime_update = b3_prime.assign(b3)

    all_assigns = [
        w1_prime_update,
        w2_prime_update,
        w3_prime_update,
        b1_prime_update,
        b2_prime_update,
        b3_prime_update]

    # First Q 3 Layered Network without dropout
    states_placeholder = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]])
    hidden_1 = tf.nn.relu(tf.matmul(states_placeholder, w1) + b1)
    hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2) + b2)
    # hidden_2 = tf.nn.dropout(hidden_2, .5)
    Q = tf.matmul(hidden_2, w3) + b3
    # Second Q 3 Layered Network without dropout
    hidden_1_prime = tf.nn.relu(tf.matmul(states_placeholder, w1_prime) + b1_prime)
    hidden_2_prime = tf.nn.relu(tf.matmul(hidden_1_prime, w2_prime) + b2_prime)
    # hidden_2_prime = tf.nn.dropout(hidden_2_prime, .5)
    Q_prime = tf.matmul(hidden_2_prime, w3_prime) + b3_prime

    action_used_placeholder = tf.placeholder(tf.int32, [None], name="action_masks")
    action_masks = tf.one_hot(action_used_placeholder, env.action_space.n)
    filtered_Q = tf.reduce_sum(tf.mul(Q, action_masks), reduction_indices=1)

    # Training Q
    target_q_placeholder = tf.placeholder(tf.float32,
                                          [None, ])  # This holds all the rewards that are real/enhanced with Qprime
    loss = tf.reduce_sum(tf.square(filtered_Q - target_q_placeholder))
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Setting up the enviroment

    D = []
    explore = 1
    rewardList = []
    past_actions = []

    episode_number = 0
    episode_reward = 0
    reward_sum = 0
    sub_goal = 0
    dir = input('Enter left or right: ')

    xmax = 1
    ymax = 1
    xind = 1
    yind = 3

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(all_assigns)

        ticks = 0
        for episode in xrange(max_episodes):
            state = env.reset()

            reward_sum = 0

            for step in xrange(max_steps):
                ticks += 1

                # print state
                xmax = max(xmax, state[xind])
                ymax = max(ymax, state[yind])

                if episode % 10 == 0:
                    q, qp = sess.run([Q, Q_prime], feed_dict={states_placeholder: np.array([state])})
                    print "Q:{}, Q_ {}".format(q[0], qp[0])
                    # print "T: {} S {}".format(ticks, state)
                    env.render()

                if explore > random.random():
                   action = env.action_space.sample()
                else:
                    # get action from policy
                    q = sess.run(Q, feed_dict={states_placeholder: np.array([state])})[0]
                    action = np.argmax(q)
                    # print action


                explore = max(explore * explore_decay, min_explore)
                action = env.action_space.sample()


                #Need to figure how to create states to go left or right
                




                #Creating subgoal here
                #Asking for left or right direction
                #If Right make the rewards higher for the right side and less for the left
                #If Left make the rewards higher for the left side and less for the right

                if dir = 'left'
                    sub_goal = sub_goal -1.0
                    new_state, reward, done, _ = env.step(action)
                    state.append(sub_goal)
                    reward_sum += reward
                if dir = 'right'
                    state.append(sub_goal)
                    new_state, reward, done, _ = env.step(action)
                    sub_goal = sub_goal +1.0
                    reward_sum += reward

                D.append([state, action, reward, new_state, done])
                if len(D) > memory_size:
                    D.pop(0);

                state = new_state

                if done:
                    break

                # Training a Batch
                samples = random.sample(D, min(batch_number, len(D)))

                # print samples

                # calculate all next Q's together for speed
                new_states = [x[3] for x in samples]
                all_q_prime = sess.run(Q_prime, feed_dict={states_placeholder: new_states})

                y_ = []
                state_samples = []
                actions = []
                terminalcount = 0
                for ind, i_sample in enumerate(samples):
                    state_mem, curr_action, reward, new_state, done = i_sample
                    if done:
                        y_.append(reward)
                        terminalcount += 1
                    else:
                        # this_q_prime = sess.run(Q_prime, feed_dict={states_placeholder: [new_state]})[0]
                        this_q_prime = all_q_prime[ind]
                        maxq = max(this_q_prime)
                        y_.append(reward + (gamma * maxq))

                    state_samples.append(state_mem)

                    actions.append(curr_action);
                sess.run([train], feed_dict={states_placeholder: state_samples, target_q_placeholder: y_,
                                             action_used_placeholder: actions})
                if ticks % num_of_ticks_between_q_copies == 0:
                    sess.run(all_assigns)

            if episode % 10 == 0:
                teststate = [0 for x in xrange(env.observation_space.shape[0])]
                # print "S: ", teststate
                X = []
                Y = []
                Z = []
                ZR = []

                xmin = -xmax
                xstep = xmax / 100.0

                ymin = -ymax
                ystep = ymax / 100.0

                test_state_list = []

            print 'Reward for episode %f is %f.' % (episode, reward_sum)

env.monitor.close()
