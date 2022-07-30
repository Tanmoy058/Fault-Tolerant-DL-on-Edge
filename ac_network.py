import numpy as np
import theano, theano.tensor as T
import lasagne
from collections import OrderedDict

class ACLearner:
    def __init__(self, pa):
        # network dimensions
        self.input_height = pa.network_input_height
        self.input_width = pa.network_input_width
        self.output_height = pa.network_output_dim

        # printing dimensions, like the PG network does
        # print 'network_input_height=', pa.network_input_height
        # print 'network_input_width=', pa.network_input_width
        # print 'network_output_dim=', pa.network_output_dim

        self.num_frames = pa.num_frames
        self.update_counter = 0

        # output layers from AC networks - returns list with [policy_layer, value_layer]
        self.output_layers = build_AC_network(self.input_height, self.input_width, self.output_height)

        # training parameters
        self.lr_rate = pa.lr_rate
        self.rms_rho = pa.rms_rho
        self.rms_eps = pa.rms_eps

        # network params
        params = lasagne.layers.helper.get_all_params(self.output_layers)
        # print ' params=', params, ' count=', lasagne.layers.count_params(self.output_layers)
        self._get_param = theano.function([], params)

        # indiv network params
        policy_params = lasagne.layers.helper.get_all_params(self.output_layers[0])
        self._get_pol_param = theano.function([], policy_params)

        value_params = lasagne.layers.helper.get_all_params(self.output_layers[1])
        self._get_val_param = theano.function([], value_params)


        # -------------------- training functions --------------------
        states = T.tensor4('states')

        actions = T.ivector('actions')

        values = T.vector('values')

        act_probabilities = lasagne.layers.get_output(self.output_layers[0], states)
        value_outputs = lasagne.layers.get_output(self.output_layers[1], states)

        self._get_act_probs = theano.function([states], act_probabilities, allow_input_downcast=True)
        self._get_values = theano.function([states], value_outputs, allow_input_downcast=True)

        N = states.shape[0]  # num of states

        # log loss of policy logits
        policy_loss = T.log(act_probabilities[T.arange(N), actions]).dot(values) / N

        # for value loss, find advantages by calculating difference in experienced value with predicted values
        # can be seen as values of performing specific action in S vs. overall prediction for value of S
        advantages = values - value_outputs
        value_loss = advantages ** 2

        # mean reduction for total loss
        loss = T.mean((policy_loss + value_loss))

        # calculates gradients
        grads = T.grad(loss, params)

        # calculate updates using RMSProp
        updates = rmsprop_updates(
            grads, params, self.lr_rate, self.rms_rho, self.rms_eps)

        # functions
        self._train_fn = theano.function([states, actions, values], loss,
                                         updates=updates, allow_input_downcast=True)

        self._get_loss = theano.function([states, actions, values], loss, allow_input_downcast=True)

        self._get_grad = theano.function([states, actions, values], grads, allow_input_downcast=True)

        # -------------------- supervised training --------------------
        su_target_act = T.ivector('su_target_act')
        su_target_val = T.vector('su_target_val')

        # cross-entropy loss from policy network
        su_policy_loss = lasagne.objectives.categorical_crossentropy(act_probabilities, su_target_act)
        su_policy_loss = su_policy_loss.mean()

        # MSE loss from value network
        su_value_loss = lasagne.objectives.squared_error(su_target_val.reshape(shape=(pa.batch_size, 1)), value_outputs)
        su_value_loss = su_value_loss.mean()

        # apply regularization penalty to both losses
        l2_penalty_policy = lasagne.regularization.regularize_network_params(self.output_layers[0], lasagne.regularization.l2)
        l2_penalty_value = lasagne.regularization.regularize_network_params(self.output_layers[1], lasagne.regularization.l2)

        su_policy_loss += (1e-3 * l2_penalty_policy)
        su_value_loss += (1e-3 * l2_penalty_value)

        # total loss
        su_loss = su_policy_loss + su_value_loss

        # print 'lr_rate=', self.lr_rate

        # update with rmsprop, find policy updates first then add value updates
        su_updates = lasagne.updates.rmsprop(su_policy_loss, policy_params, self.lr_rate, self.rms_rho, self.rms_eps)
        su_updates.update(lasagne.updates.rmsprop(su_value_loss, value_params, self.lr_rate, self.rms_rho, self.rms_eps))

        self._su_train = theano.function([states, su_target_act, su_target_val],
                                         [su_loss, act_probabilities, value_outputs],
                                         updates=su_updates)

        self._su_loss = theano.function([states, su_target_act, su_target_val],
                                        [su_loss, act_probabilities, value_outputs])

        self._debug = theano.function([states], [states.flatten(2)])

    # choose an action given the state
    def choose_action(self, state):
        # get all action probabilities
        act_prob = self.get_one_act_prob(state)

        # cumulative sum of probabilities
        csprob_n = np.cumsum(act_prob)

        # weird way of doing a choice based on probs?
        act = (csprob_n > np.random.rand()).argmax()

        # print(act_prob, act)
        return act

    # returns list of action probabilities summing to 1
    def get_one_act_prob(self, state):

        states = np.zeros((1, 1, self.input_height, self.input_width), dtype=theano.config.floatX)
        states[0, :, :] = state
        act_prob = self._get_act_probs(states)[0]

        return act_prob

    # multiple states, assuming in floatX format
    def get_act_probs(self, states):
        act_probs = self._get_act_probs(states)
        return act_probs

    # --- reinforcement training helpers same as PG network ---
    def train(self, states, actions, values):
        loss = self._train_fn(states, actions, values)
        return loss

    def get_params(self):
        return self._get_param()

    def get_grad(self, states, actions, values):
        return self._get_grad(states, actions, values)

    #  -------- Supervised Learning --------
    def su_train(self, states, target_acts, target_rews):
        loss, prob_act, vals = self._su_train(states, target_acts, target_rews)
        return np.sqrt(loss), prob_act, vals

    def su_test(self, states, target_acts, target_vals):
        loss, prob_act, vals = self._su_loss(states, target_acts, target_vals)
        return np.sqrt(loss), prob_act, vals

    #  -------- Save/Load network parameters --------
    # have to return params as a tuple of policy and value params
    def return_net_params(self):
        policy_out, value_out = self.output_layers[0], self.output_layers[1]
        return lasagne.layers.helper.get_all_param_values(policy_out), lasagne.layers.helper.get_all_param_values(value_out)

    # have to set policy params and value params independently
    def set_net_params(self, policy_net_params, value_net_params):
        policy_out, value_out = self.output_layers[0], self.output_layers[1]
        lasagne.layers.helper.set_all_param_values(policy_out, policy_net_params)
        lasagne.layers.helper.set_all_param_values(value_out, value_net_params)


# don't fully understand RMSProp optimization for mini-batching, but this shouldn't need to be altered
def rmsprop_updates(grads, params, stepsize, rho=0.9, epsilon=1e-9):
    updates = []

    for param, grad in zip(params, grads):
        accum = theano.shared(np.zeros(param.get_value(borrow=True).shape, dtype=param.dtype))
        accum_new = rho * accum + (1 - rho) * grad ** 2
        updates.append((accum, accum_new))
        updates.append((param, param + (stepsize * grad / T.sqrt(accum_new + epsilon))))
        # lasagne has '-' after param
    return updates


def build_AC_network(input_height, input_width, output_dim):
    # input layers for both networks
    policy_in = lasagne.layers.InputLayer(shape=(None, input_height, input_width))
    value_in = lasagne.layers.InputLayer(shape=(None, input_height, input_width))

    # densely connected hidden policy layer
    policy_hid = lasagne.layers.DenseLayer(
        policy_in,                                      # layer passing into this one
        num_units=50,                                   # num nodes in layer
        nonlinearity=lasagne.nonlinearities.rectify,    # ReLU activation
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(0)
    )

    # densely connected hidden value layer
    value_hid = lasagne.layers.DenseLayer(
        value_in,
        num_units=50,                                   # num nodes in layer
        nonlinearity=lasagne.nonlinearities.rectify,    # ReLU activation
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(0)
    )

    policy_out = lasagne.layers.DenseLayer(
        policy_hid,
        num_units=output_dim,                           # output dim is action space
        nonlinearity=lasagne.nonlinearities.softmax,    # soft-max activation for action probabilities
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(0)
    )

    value_out = lasagne.layers.DenseLayer(
        value_hid,
        num_units=1,                                    # 1 linearly activated output
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(0)
    )

    return [policy_out, value_out]


