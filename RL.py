import os
import threading
import gym
import multiprocessing
import numpy as np
from queue import Queue
import argparse
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python import keras

os.environ["CUDA_VISIBLE_DEVICES"] = ""
tf.enable_eager_execution()

# argument parsing
parser = argparse.ArgumentParser(description='Run A3C algorithm on the game '
                                             'Cartpole.')
parser.add_argument('--algorithm', default='a3c', type=str,
                    help='Choose between \'a3c\' and \'random\'.')
parser.add_argument('--train', dest='train', action='store_true',
                    help='Train our model.')
parser.add_argument('--lr', default=0.001,
                    help='Learning rate for the shared optimizer.')
parser.add_argument('--update-freq', default=20, type=int,
                    help='How often to update the global model.')
parser.add_argument('--max-eps', default=1000, type=int,
                    help='Global maximum number of episodes to run.')
parser.add_argument('--gamma', default=0.99,
                    help='Discount factor of rewards.')
parser.add_argument('--save-dir', default='/tmp/', type=str,
                    help='Directory in which you desire to save the model.')
parser.add_argument('--beta', default=.01, type=float,
                    help='Weight of entropy in loss')
parser.add_argument('--trials', default=1, type=int,
                    help='Trials for playing (ignored in train mode')
args = parser.parse_args()


# for easy print statements
def log(episode, ep_reward, workerid, step_count, average_reward, total_loss):
    line = "Episode: {} Average Reward: {} Episode Reward: {} Loss: {} Steps: {} Worker: {}"
    print(line.format(episode, average_reward, ep_reward, total_loss, step_count, workerid))


class RandomBaseline:
    """
        A random agent to establish baseline performance
    """

    def __init__(self, env_name, max_eps):
        self.env = gym.make(env_name)
        self.max_eps = max_eps
        self.moving_average = np.NINF
        self.results = Queue()

    def run(self):
        total_reward_sum = 0
        for episode in range(self.max_eps):
            steps = 0
            done = False
            ep_reward_sum = 0
            self.env.reset()
            while not done:
                action = self.env.action_space.sample()
                _, reward, done, _ = self.env.step(action=action)
                steps += 1
                ep_reward_sum += reward

            if self.moving_average == np.NINF:
                self.moving_average = ep_reward_sum
            else:
                self.moving_average = .99 * self.moving_average + .01 * ep_reward_sum
            self.results.put(self.moving_average)

            log(episode, ep_reward_sum, 0, steps, self.moving_average, 0)

            total_reward_sum += ep_reward_sum
        average_reward = total_reward_sum / self.max_eps
        print("Average score of {} across {} episodes".format(average_reward, self.max_eps))
        return average_reward


class ActorCriticModel(keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCriticModel, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        # the policy network
        self.policy_dense1 = tf.layers.Dense(5, activation="relu")
        self.policy_dense2 = tf.layers.Dense(10, activation="relu")
        self.policy_dense3 = tf.layers.Dense(25, activation="relu")
        # self.policy_dense4 = tf.layers.Dense(40, activation="relu")
        self.policy_output = tf.layers.Dense(action_size)

        # the state-value network
        self.value_dense1 = tf.layers.Dense(5, activation="relu")
        self.value_dense2 = tf.layers.Dense(10, activation="relu")
        self.value_dense3 = tf.layers.Dense(25, activation="relu")
        # self.value_dense4 = tf.layers.Dense(40, activation="relu")
        self.value_output = tf.layers.Dense(1)              # no activation (a(x) = x)

    def call(self, input):
        # a forward pass through each network
        x1 = self.policy_dense1(input)
        x2 = self.policy_dense2(x1)
        x3 = self.policy_dense3(x2)
        policy_vals = self.policy_output(x3)

        y1 = self.value_dense1(input)
        y2 = self.value_dense2(y1)
        y3 = self.value_dense3(y2)
        #  = self.value_dense4(y3)
        value = self.value_output(y3)

        return policy_vals, value

class MasterAgent():
    def __init__(self):
        # environment stuff
        self.game = "LunarLander-v2"
        env = gym.make(self.game)
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        print("State Size: {} Action Size: {}".format(self.state_size, self.action_size))

        # model saving
        self.save_dir = args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # optimizer and model
        self.opt = tf.train.AdamOptimizer(float(args.lr), use_locking=True)

        self.global_model = ActorCriticModel(self.state_size, self.action_size)
        # initialize
        self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

    def train(self):
        if args.algorithm == "random":
            random_agent = RandomBaseline(self.game, args.max_eps)
            random_agent.run()
            return

        result_queue = Queue()  # track results of all workers

        # create workers for each cpu
        workers = []
        for i in range(multiprocessing.cpu_count()):
            workers.append(Worker(self.state_size, self.action_size, self.global_model, self.opt, result_queue,
                           i, self.game, self.save_dir))

        # start each worker
        for i in range(len(workers)):
            workers[i].start()
            print("Started worker {}".format(i))

        average_over_time = []  # for graphing
        # pull from results until None
        while True:
            rew = result_queue.get()
            if rew is not None:
                average_over_time.append(rew)
            else:
                break

        # close the threads
        for worker in workers:
            worker.join()

        # plot average over time
        plt.plot(average_over_time)
        plt.xlabel = "episode"
        plt.ylabel = "average reward"
        plt.show()

    def play(self):
        trials = 0
        total_reward = 0
        # load our trained model
        model = self.global_model
        model_path = os.path.join(self.save_dir, 'model_{}.h5'.format(self.game))
        print('Loading model from: {}'.format(model_path))
        model.load_weights(model_path)

        env = gym.make(self.game).unwrapped
        while trials < args.trials:
            state = env.reset()
            # state = np.append(state, 0)

            # state of play
            done = False
            step_counter = 0
            reward_sum = 0
            reward_freq = {}
            steps = 0
            try:
                while not done:
                    if args.trials == 1:
                        env.render(mode='rgb_array')  # show environment

                    # convert current state to tensor, taking None Rows and all columns? (maybe 1 dimensional data)
                    # using our actor-critic call method, return policy logits and value
                    policy, value = model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))

                    # softmax activation gives probabilities summing to 1
                    policy = tf.nn.softmax(policy)

                    # index of max probability gives our action
                    action = np.argmax(policy)

                    # perform the action. get the state, reward (1 if still alive), and if terminated in return.
                    state, reward, done, _ = env.step(action)
                    # state = np.append(state, steps)
                    steps += 1

                    if steps == 1501:
                        reward = -100
                        done = True

                    if reward in reward_freq.keys():
                        reward_freq[reward] += 1
                    else:
                        reward_freq[reward] = 1
                    reward_sum += reward  # tracking reward for the play

                    # logging
                    print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
                    step_counter += 1
            except KeyboardInterrupt:
                print("Received Keyboard Interrupt. Shutting down.")
            finally:
                total_reward += reward_sum
                trials += 1
                env.close()

                # plot specific rewards received and their freq
                if (args.trials == 1):
                    plt.bar(reward_freq.keys(), reward_freq.values())
                    plt.xlabel = "Reward Value"
                    plt.ylabel = "Freq Reward was Received"
                    plt.show()

        print("Testing Averaged {} Reward Across {} Trials".format(total_reward / args.trials, args.trials))


class Worker(threading.Thread):
    # global variables for tracking overall training state
    global_episode = 0
    global_average = np.NINF
    best_score = np.NINF

    # thread locking for shared resources
    lock = threading.Lock()

    def __init__(self, state_size, action_size, global_model, opt, result_queue, workerID, game, save_dir):
        super(Worker, self).__init__()
        # environment
        self.state_size = state_size
        self.action_size = action_size
        self.game = game
        self.env = gym.make(game).unwrapped

        # global model and optimizer
        self.global_model = global_model
        self.opt = opt

        # globally shared result queue
        self.result_queue = result_queue

        self.workerID = workerID
        self.save_dir = save_dir

        # local model
        self.local_model = ActorCriticModel(state_size, action_size)

    def run(self):
        # for easy storage of each episode
        ep_mem = Memory()

        while Worker.global_episode < args.max_eps:
            # episode state
            ep_loss = 0
            ep_reward = 0
            ep_steps = 0

            # environment
            current_state = self.env.reset()
            # current_state = np.append(current_state, ep_steps)

            time_passed = 0
            terminated = False
            while not terminated:
                # perform action based on local model policy
                policy_outputs, _ = self.local_model(tf.convert_to_tensor(current_state[None, :], dtype=tf.float32))
                action_probs = tf.nn.softmax(policy_outputs)

                # sample range (0, action_size) with prob dist given by policy
                action = np.random.choice(self.action_size, p=action_probs.numpy()[0])

                # perform the action
                new_state, reward, terminated, _ = self.env.step(action)

                ep_steps += 1
                # new_state = np.append(new_state, ep_steps)

                # consecutive negative actions = worse reward
                if len(ep_mem.rewards) > 0:
                    if (ep_mem.rewards[-1] < 0) & (reward < 0):
                        reward = 2 * reward

                if ep_steps > 1500:
                    ep_reward -= 500
                    terminated = True

                # update state
                ep_reward += reward
                ep_mem.update(current_state, action, reward)

                # periodically update, push to global network
                if (time_passed == args.update_freq) | terminated:
                    # find loss
                    with tf.GradientTape() as tape:
                        total_loss = self.loss(new_state, terminated, ep_mem, float(args.gamma))
                    ep_loss += total_loss

                    # calculate gradients
                    local_gradients = tape.gradient(total_loss, self.local_model.trainable_weights)
                    # push local gradients to global, pull new weights to local model
                    self.opt.apply_gradients(zip(local_gradients, self.global_model.trainable_weights))
                    self.local_model.set_weights(self.global_model.get_weights())

                    # reset periodic update stuff
                    ep_mem.clear()
                    time_passed = 0

                    # log if terminated, saving model if best model so far
                    if terminated:
                        if Worker.global_average == np.NINF:
                            Worker.global_average = ep_reward
                        else:
                            Worker.global_average = .99 * Worker.global_average + .01 * ep_reward
                        self.result_queue.put(Worker.global_average)
                        log(Worker.global_episode, ep_reward, self.workerID, ep_steps, Worker.global_average, ep_loss)

                        # save if best score
                        # | (Worker.global_episode % 500 == 0)
                        if ep_reward > Worker.best_score:
                            with self.lock:
                                # save the global model
                                print("Saving best model to {}, acheived score: {}".format(self.save_dir, ep_reward))
                                self.global_model.save_weights(
                                    os.path.join(self.save_dir,
                                                 'model_{}.h5'.format(self.game))
                                )
                                if (ep_reward > Worker.best_score):
                                    Worker.best_score = ep_reward
                        # global training state update
                        Worker.global_episode += 1

                # local episode state
                self.env.close()
                time_passed += 1
                current_state = new_state
        # terminate signal for master training
        self.result_queue.put(None)

    def loss(self, new_state, terminated, mem, gamma):
        # predicted future reward, V(next state)
        if terminated:
            reward_sum = 0.  # terminal
        else:
            # get second thing (reward) in tuple from passing new state
            reward_sum = self.local_model(
                tf.convert_to_tensor(new_state[None, :],
                                     dtype=tf.float32))[-1].numpy()[0]

        # Get discounted rewards
        discounted_rewards = []
        for reward in mem.rewards[::-1]:  # reverse buffer r, to look at most recent (latest rewards) first
            # Q value of the current state based on V, (R(t+1) + V(next state))
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        # ie. first element is Q val of s0, second of s1, ...
        # we now have the estimated Q value of each state action pair in this episode

        # array of logits and values at each remembered state
        # values are the state-value (v(s)) for each state observed in this episode
        logits, values = self.local_model(
            tf.convert_to_tensor(np.vstack(mem.states),
                                 dtype=tf.float32))

        # Get our advantages
        # discounted cumulative reward of each (s, a) - learned value of this state, Q(s, a) - V(s)
        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                                         dtype=tf.float32) - values

        # Value loss
        value_loss = advantage ** 2

        # Calculate our policy loss
        policy = tf.nn.softmax(logits)  # probabilities for each action at every state we were in
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)

        # the loss of this episode policy, using each action picked and the policy network output of those actions
        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=mem.actions,
                                                                     logits=logits)
        policy_loss *= tf.stop_gradient(advantage)  # stop gradient stops advantages from affecting gradient?

        # encourage exploration, we don't want the model to become very confident in certain actions
        # since then it won't discover actions that leader to greater cumm reward
        policy_loss -= args.beta * entropy

        # total loss is the mean of (.5 * value loss at each step + policy loss at each step)
        total_loss = tf.reduce_mean((value_loss + policy_loss))
        return total_loss


# class for easy storing of episode data
class Memory:
    def __init__(self):
        self.states = []    # the state, S
        self.actions = []   # the action, a
        self.rewards = []   # reward from performing a in S

    def update(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


if __name__ == '__main__':
    print(args)
    master = MasterAgent()
    if args.train:
        master.train()
    else:
        master.play()
