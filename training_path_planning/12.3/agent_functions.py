import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers

def create_actor(args):
    # initialize last layer weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-3, maxval=3)
    # layer_init = tf.keras.initializers.Zeros()
    # layer_init = initializers.zeros()

    # define net layers
    inputs = layers.Input(shape=(args.state_size, args.state_size, 1))
    out = layers.Conv2D(16, (7, 7), input_shape=(args.state_size, args.state_size, 1), strides=(1,1), activation="relu", padding='valid')(inputs)
    out = layers.MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='same')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Conv2D(32, (5, 5), strides=(3, 3), activation="relu", padding='valid')(out)
    out = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Conv2D(64, (3, 3), strides=(2, 2), activation="relu", padding='valid')(out)
    out = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Flatten()(out)
    # out = layers.Dropout(.2)(out)
    # out = layers.Dense(128, activation="relu", kernel_initializer=layer_init)(out)
    out = layers.Dense(128, activation="relu")(out)
    out = layers.Dense(64, activation="relu")(out)
    out = layers.Dense(32, activation="relu")(out)
    out = layers.Dense(16, activation="relu")(out)
    out = layers.Dense(8, activation="relu")(out)
    # out = layers.LayerNormalization()(out)
    outputs = layers.Dense(1, activation="sigmoid", kernel_initializer=last_init)(out)

    # rescale output to match action upper and lower bound
    outputs = outputs * math.radians(args.max_angle)

    # build model
    model = tf.keras.Model(inputs, outputs)
    return model


def create_critic(args):
    # state input + part of net that processes state
    state_input = layers.Input(shape=(args.state_size, args.state_size, 1))
    out = layers.Conv2D(16, (7, 7), input_shape=(args.state_size, args.state_size, 1), activation="relu")(state_input)
    out = layers.MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='same')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Conv2D(32, (5, 5), strides=(3, 3), activation="relu")(out)
    out = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Flatten()(out)
    # out = layers.Dense(64, activation="relu")(out)
    state_out = layers.Dense(64, activation="relu")(out)

    # action input + part of net that processes action
    action_input = layers.Input(shape=args.num_actions)
    action_out = layers.Dense(32, activation="relu")(action_input)
    action_out = layers.Dense(16, activation="relu")(action_out)
    # action_layer = layers.Dense(16, activation="relu")(action_input)
    # action_layer = layers.Dense(8, activation="relu")(action_layer)
    # action_out = layers.Dense(1, activation="relu")(action_layer)

    # concatenation of the two inputs
    concat = layers.Concatenate()([state_out, action_out])

    # part of net that processes the concatenated inputs
    out = layers.Dense(128, activation="relu")(concat)
    out = layers.Dense(64, activation="relu")(out)
    out = layers.Dense(32, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # build model - output is a single Q-value for give state-action couple
    model = tf.keras.Model([state_input, action_input], outputs)
    return model

def policy(actor_model, state, noise_object, args):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()

    # adding noise to action (for exploration)
    sampled_actions = sampled_actions.numpy() + noise

    # make sure action is within bounds (due to noise presence)
    legal_action = np.clip(sampled_actions, math.radians(args.min_angle), math.radians(args.max_angle))
    return [np.squeeze(legal_action)]

# smoothed target update - this update target parameters slowly, based on rate `tau`, which is much less than one
@tf.function
def update_target(target_weights, weights, smoothing_factor_tau):
    for (target, net) in zip(target_weights, weights):
        target.assign(net * smoothing_factor_tau + target * (1 - smoothing_factor_tau))