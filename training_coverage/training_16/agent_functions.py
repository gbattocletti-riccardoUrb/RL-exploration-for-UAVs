# import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def create_actor(args):
    # initialize last layer weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-3, maxval=3)
    # layer_init = tf.keras.initializers.Zeros()
    # layer_init = initializers.zeros()

    # define net layers
    image_input = layers.Input(shape=(args.state_size, args.state_size, 1))
    out = layers.Conv2D(32, (5, 5), input_shape=(args.state_size, args.state_size, 1), strides=(2, 2), activation="relu", padding='valid')(image_input)
    out = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Conv2D(32, (5, 5), strides=(2, 2), activation="relu", padding='valid')(out)
    out = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Conv2D(64, (3, 3), strides=(2, 2), activation="relu", padding='valid')(out)
    out = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(out)
    out = layers.BatchNormalization()(out)
    image_out = layers.Flatten()(out)
    image_out = layers.Dropout(0.2)(image_out)

    goal_distance_input = layers.Input(shape=2)
    out = layers.Dense(256, activation="relu")(goal_distance_input)
    out = layers.Dense(128, activation="relu")(out)
    out = layers.Dense(64, activation="relu")(out)
    out = layers.Dense(32, activation="relu")(out)
    out = layers.Dense(16, activation="relu")(out)
    goal_out = layers.Dense(8, activation="relu")(out)
    # out = layers.Dropout(.2)(out)
    # out = layers.Dense(128, activation="relu", kernel_initializer=layer_init)(out)
    concat = layers.Concatenate()([image_out, goal_out])
    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(128, activation="relu")(out)
    out = layers.Dense(64, activation="relu")(out)
    out = layers.Dense(32, activation="relu")(out)
    out = layers.Dense(16, activation="relu")(out)
    out = layers.Dense(8, activation="relu")(out)
    # out = layers.LayerNormalization()(out)
    outputs = layers.Dense(2, activation="sigmoid", kernel_initializer=last_init)(out)
    # rescale output to match action upper and lower bound
    # outputs = outputs * args.max_output

    # build model
    model = tf.keras.Model([image_input, goal_distance_input], outputs)
    return model


def create_critic(args):
    # state input + part of net that processes state
    state_input = layers.Input(shape=(args.state_size, args.state_size, 1))
    out = layers.Conv2D(32, (5, 5), strides=(2, 2), input_shape=(args.state_size, args.state_size, 1), activation="relu")(state_input)
    out = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Conv2D(32, (5, 5), strides=(2, 2), activation="relu")(out)
    out = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Conv2D(64, (3, 3), strides=(2, 2), activation="relu", padding='valid')(out)
    out = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Flatten()(out)
    # out = layers.Dropout(0.2)(out)
    # out = layers.Dense(64, activation="relu")(out)
    out = layers.Dense(256, activation="relu")(out)
    out = layers.Dense(128, activation="relu")(out)
    out = layers.Dense(64, activation="relu")(out)
    out = layers.Dense(32, activation="relu")(out)
    state_out = layers.Dense(16, activation="relu")(out)

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
    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(128, activation="relu")(out)
    out = layers.Dense(64, activation="relu")(out)
    out = layers.Dense(32, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # build model - output is a single Q-value for give state-action couple
    model = tf.keras.Model([state_input, action_input], outputs)
    return model

def policy(actor_model, state, noise_object, goal_distance):
    sampled_actions = tf.squeeze(actor_model([state, goal_distance]))
    noise = noise_object()

    # adding noise to action (for exploration)
    sampled_actions = sampled_actions.numpy() + noise

    # make sure action is within bounds (due to noise presence)
    legal_action = np.clip(sampled_actions, 0, 1)
    return [np.squeeze(legal_action)]

# smoothed target update - this update target parameters slowly, based on rate `tau`, which is much less than one
@tf.function
def update_target(target_weights, weights, smoothing_factor_tau):
    for (target, net) in zip(target_weights, weights):
        target.assign(net * smoothing_factor_tau + target * (1 - smoothing_factor_tau))