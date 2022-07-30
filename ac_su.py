import numpy as np
import theano
import time
import sys
import cPickle

import environment
import ac_network
import other_agents
import job_distribution


# need to add samples at end of each episode instead, since we need cumulative discounted reward measures
def add_episode_samples(X, y_acts, y_vals, episode_start, episode_lenth, states_to_add, acts_to_add, rews_to_add):
    # need to convert the individual rewards to cumulative discounted
    cum_rew = 0
    for i in range(len(rews_to_add) - 1, -1, -1):
        cum_rew = rews_to_add[i] + .99*cum_rew
        rews_to_add[i] = cum_rew

    # now add to data
    X[episode_start:episode_start+episode_lenth, 0, :, :] = states_to_add[0:episode_lenth]
    y_acts[episode_start:episode_start+episode_lenth] = acts_to_add[0:episode_lenth]
    y_vals[episode_start:episode_start+episode_lenth] = rews_to_add[0:episode_lenth]


def iterate_minibatches(inputs, target_acts, target_rews, batchsize, shuffle=False):
    assert len(inputs) == len(target_acts)
    assert len(inputs) == len(target_rews)

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], target_acts[excerpt], target_rews[excerpt]


# should be able to use pretty much same setup as in pg_su, with addition of values in target trials
def launch(pa, pg_resume=None, render=False, repre='image', end='no_new_job'):

    # env and model
    env = environment.Env(pa, render=False, repre=repre, end=end)
    ac_learner = ac_network.ACLearner(pa)

    # resume training, load networks
    if pg_resume is not None:
        # load and set individual networks
        policy_net_handle = open(pg_resume + "_pol.pkl", 'r')
        value_net_handle = open(pg_resume + "_val.pkl", 'r')
        policy_net_params = cPickle.load(policy_net_handle)
        value_net_params = cPickle.load(value_net_handle)
        ac_learner.set_net_params(policy_net_params, value_net_params)

    # scheduling policies to perform supervised learning with
    if pa.evaluate_policy_name == "SJF":
        evaluate_policy = other_agents.get_sjf_action
    elif pa.evaluate_policy_name == "PACKER":
        evaluate_policy = other_agents.get_packer_action
    else:
        print("Panic: no policy known to evaluate.")
        exit(1)

    # ----------------------------
    print("Preparing for data...")
    # ----------------------------

    nw_len_seqs, nw_size_seqs = job_distribution.generate_sequence_work(pa, seed=42)

    # print 'nw_time_seqs=', nw_len_seqs
    # print 'nw_size_seqs=', nw_size_seqs

    mem_alloc = 4

    # need additional y to track actions and values
    X = np.zeros([pa.simu_len * pa.num_ex * mem_alloc, 1,
                  pa.network_input_height, pa.network_input_width],
                 dtype=theano.config.floatX)
    y_acts = np.zeros(pa.simu_len * pa.num_ex * mem_alloc,
                 dtype='int32')
    y_rews = np.zeros(pa.simu_len * pa.num_ex * mem_alloc,
                 dtype=theano.config.floatX)

    print 'network_input_height=', pa.network_input_height
    print 'network_input_width=', pa.network_input_width

    counter = 0

    # generate state action pairs for the number of examples needed
    # need cumulative discounted reward for each state, so need to wait for episode termination before adding
    for train_ex in range(pa.num_ex):

        env.reset()

        episode_states = np.zeros((pa.episode_max_length, pa.network_input_height, pa.network_input_width),
                                  dtype=theano.config.floatX)
        episode_actions = np.zeros(pa.episode_max_length, dtype='int32')
        episode_rewards = np.zeros(pa.episode_max_length, dtype=theano.config.floatX)

        episode_length = 0
        episode_start = counter
        for i in xrange(pa.episode_max_length):

            # ---- get current state ----
            ob = env.observe()

            a = evaluate_policy(env.machine, env.job_slot)

            ob, rew, done, info = env.step(a, repeat=True)

            if counter < pa.simu_len * pa.num_ex * mem_alloc:
                episode_states[i] = ob
                episode_actions[i] = a
                episode_rewards[i] = rew
                i += 1
                episode_length += 1
                counter += 1

            if done:  # hit void action, exit
                break

        add_episode_samples(X, y_acts, y_rews, episode_start, episode_length,
                            episode_states, episode_actions, episode_rewards)

        # roll to next example
        env.seq_no = (env.seq_no + 1) % env.pa.num_ex

    # 80 / 20 testing train split
    num_train = int(0.8 * counter)
    num_test = int(0.2 * counter)

    X_train, X_test = X[:num_train], X[num_train: num_train + num_test]
    y_acts_train, y_acts_test = y_acts[:num_train], y_acts[num_train: num_train + num_test]
    y_rews_train, y_rews_test = y_rews[:num_train], y_rews[num_train: num_train + num_test]

    # Normalization, make sure nothing becomes NaN

    # X_mean = np.average(X[:num_train + num_test], axis=0)
    # X_std = np.std(X[:num_train + num_test], axis=0)

    # X_train = (X_train - X_mean) / X_std
    # X_test = (X_test - X_mean) / X_std

    # ----------------------------
    print("Start training...")
    # ----------------------------

    # pretty much same as PG network, except need training batches to include reward
    # also need to save 2 networks instead of 1
    for epoch in xrange(pa.num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_acts_train, y_rews_train, pa.batch_size, shuffle=True):
            inputs, target_acts, target_rews = batch
            #print("Shape of inputs batch: " + str(inputs.shape))
            #print("Shape of acts batch: " + str(target_acts.shape))
            #print("Shape of rews batch: " + str(target_rews.shape))
            err, prob_act, _ = ac_learner.su_train(inputs, target_acts, target_rews)
            ac_act = np.argmax(prob_act, axis=1)
            train_err += err
            train_acc += np.sum(ac_act == target_acts)
            train_batches += 1

        # # And a full pass over the test data:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, y_acts_test, y_rews_test, pa.batch_size, shuffle=False):
            inputs, target_acts, target_rews = batch
            err, prob_act, _ = ac_learner.su_test(inputs, target_acts, target_rews)
            ac_act = np.argmax(prob_act, axis=1)
            test_err += err
            test_acc += np.sum(ac_act == target_acts)
            test_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, pa.num_epochs, time.time() - start_time))
        print("  training loss:    \t\t{:.6f}".format(train_err / train_batches))
        print("  training accuracy:\t\t{:.2f} %".format(
            train_acc / float(num_train) * 100))
        print("  test loss:        \t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:    \t\t{:.2f} %".format(
            test_acc / float(num_test) * 100))

        sys.stdout.flush()

        if epoch % pa.output_freq == 0:
            params = ac_learner.return_net_params()

            pol_net_file = open(pa.output_filename + 'AC_su_net_file_' + str(epoch) + "_policy" + '.pkl', 'wb')
            cPickle.dump(params[0], pol_net_file, -1)
            pol_net_file.close()

            val_net_file = open(pa.output_filename + 'AC_su_net_file_' + str(epoch) + "_value" + '.pkl', 'wb')
            cPickle.dump(params[1], val_net_file, -1)
            val_net_file.close()

    print("done")