import numpy as np
import tensorflow as tf

class Buffer:
    def __init__(self, args):
        self.buffer_capacity = args.buffer_capacity         # number of "experiences" to store at max
        self.batch_size = args.batch_size                   # num of experiences to train on at each training step
        self.buffer_counter = 0                             # tells us num of times record() was called (to avoid exceding max dimension)

        # instead of list of tuples as the exp.replay concept goes, we use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, args.state_size, args.state_size))
        self.action_buffer = np.zeros((self.buffer_capacity, args.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, args.state_size, args.state_size))

        # choose optimizers
        self.critic_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(args.actor_lr)

    def record(self, obs_tuple):
        # takes (s,a,r,s') obervation tuple as input and stores the new experience into the memory buffer

        index = self.buffer_counter % self.buffer_capacity  # set index to zero if buffer_capacity is exceeded replacing old records

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows TensorFlow to build a
    # static graph interpolated_output of the logic and computations in our function. This provides a large speed up for blocks of code
    # that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, actor_model, critic_model, target_actor, target_critic, gamma):
        # Training and updating Actor & Critic networks.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic([next_state_batch, target_actions], training=True)
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

    # perform one learning step (one backproagation i.e. one step of gradient descent
    def learn(self, actor_model, critic_model, target_actor, target_critic, args):

        # get sampling range (min and max index of available memories)
        record_range = min(self.buffer_counter, self.buffer_capacity)

        # randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        # perform one backpropagation using the sampled minibatch - compute the loss and update parameters
        self.update(state_batch, action_batch, reward_batch, next_state_batch, actor_model, critic_model, target_actor, target_critic, args.gamma)