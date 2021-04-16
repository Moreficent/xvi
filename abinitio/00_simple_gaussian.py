"""
Calibrates a gaussian with an unknown location and known scale to data. The mean
is assumed to be normally distributed, with a prior location of 0 and prior scale
of 10, i.e. a "flat" prior. 

For simplicity, eager mode evaluation is used. 
"""
from abinitio.utils import fix_seed

fix_seed(42)

import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

np.set_printoptions(precision=2)

DATA_SCALE = 0.3
PRIOR_LOC = 0
PRIOR_SCALE = 10
PRIOR_SAMPLES = 100
MODEL_SAMPLES = 4


@tf.function
def train_step(mu, omega, prior_dist, data, optim):
    with tf.GradientTape() as tape:
        eta = tf.random.normal([1, MODEL_SAMPLES])
        theta = mu + eta * tf.math.exp(omega)
        dist = tfp.distributions.Normal(loc=theta, scale=DATA_SCALE)
        lp = dist.log_prob(data)
        lp = tf.reduce_mean(lp, axis=1)  # averaging out sample
        lp = tf.reduce_sum(lp)  # adding across samples
        elbo = lp + omega + tf.reduce_mean(prior_dist.log_prob(theta))
        loss = -elbo

    model_parameters = [mu, omega]
    gradients = tape.gradient(loss, model_parameters)
    optim.apply_gradients(zip(gradients, model_parameters))
    return loss


if __name__ == '__main__':
    # Generating the data
    data_loc = -3
    data_samples = 100
    data = tf.random.normal(shape=[data_samples, 1], mean=data_loc, stddev=DATA_SCALE)

    # Analytical Posterior
    data_mean = tf.reduce_mean(data)
    prior_var = PRIOR_SCALE * PRIOR_SCALE
    data_var = DATA_SCALE * DATA_SCALE / data_samples
    posterior_variance = 1.0 / (1.0 / prior_var + 1.0 / data_var)
    posterior_mean = posterior_variance * (PRIOR_LOC / prior_var + data_mean / data_var)

    # constructing the model
    prior_samples = tf.random.normal(shape=[PRIOR_SAMPLES], mean=PRIOR_LOC, stddev=PRIOR_SCALE)
    init_mu = tf.math.reduce_mean(prior_samples)
    init_omega = tf.math.log(tf.math.reduce_std(prior_samples))
    mu = tf.Variable(initial_value=init_mu, trainable=True, name='mu')
    omega = tf.Variable(initial_value=init_omega, trainable=True, name='omega')
    print(f'Initial parameters: mu: {init_mu:1.3f}, omega: {init_omega:1.3f}')
    print(f'Target parameters : mu: {posterior_mean:1.3f}, omega: {0.5 * math.log(posterior_variance):1.3f}')
    # # training the model
    optim = tf.keras.optimizers.RMSprop(learning_rate=5e-2)
    num_epochs = 2000
    prior_dist = tfp.distributions.Normal(loc=PRIOR_LOC, scale=PRIOR_SCALE)
    for idx in range(num_epochs):
        loss = train_step(mu, omega, prior_dist, data, optim)
        loss = loss.numpy()
        if idx % 100 == 0:
            print(f'[{idx:04d}]: loss: {loss:4.3f}, mu: {mu.numpy():1.3f}, omega: {omega.numpy():1.3f}')

    print(f'[{num_epochs:04d}]: loss: {loss:4.3f}, mu: {mu.numpy():1.3f}, omega: {omega.numpy():1.3f}')

    print('ho gaya')
