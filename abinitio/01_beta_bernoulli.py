"""
Calibrates a bernoulli distribution to data. The success probability is assumed to be 
beta distributed. 

For simplicity, eager mode evaluation is used. 
"""
from abinitio.utils import fix_seed

fix_seed(42)

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

np.set_printoptions(precision=2)

MODEL_SAMPLES = 4
PRIOR_ALPHA = 20
PRIOR_BETA = 30
PRIOR_SAMPLES = 100


@tf.function
def logit(x):
    return tf.math.log((x + 1e-8) / (1e-8 + 1 - x))


@tf.function
def train_step(mu, omega, prior_dist, data, optim):
    with tf.GradientTape() as tape:
        eta = tf.random.normal([1, MODEL_SAMPLES])
        zeta = mu + eta * tf.math.exp(omega)

        dist = tfp.distributions.Bernoulli(logits=zeta)
        lp = dist.log_prob(data)
        lp = tf.reduce_mean(lp, axis=1)
        lp = tf.reduce_sum(lp)

        theta = tf.math.sigmoid(zeta)
        prior_prob = prior_dist.log_prob(theta)
        jac = tf.math.log(theta) + tf.math.log(1 - theta)
        constraint = tf.math.reduce_mean(prior_prob + jac) + omega

        elbo = lp + constraint
        loss = -elbo

    model_parameters = [mu, omega]
    gradients = tape.gradient(loss, model_parameters)
    optim.apply_gradients(zip(gradients, model_parameters))

    return loss


if __name__ == '__main__':
    # Generating the data
    data_prob = 0.65
    data_samples = 50
    data = tf.random.stateless_binomial(shape=[data_samples, 1], seed=[3141, 2718], counts=1, probs=data_prob)

    # analytical posterior
    posterior_alpha = float(PRIOR_ALPHA + tf.math.reduce_sum(data))
    posterior_beta = float(PRIOR_BETA + data_samples - tf.math.reduce_sum(data))
    posterior_dist = tfp.distributions.Beta(posterior_alpha, posterior_beta)
    posterior_theta = posterior_dist.sample(PRIOR_SAMPLES)
    posterior_zeta = logit(posterior_theta)
    target_mu = tf.math.reduce_mean(posterior_zeta)
    target_omega = tf.math.log(tf.math.reduce_std(posterior_zeta))

    # constructing the model
    prior_dist = tfp.distributions.Beta(PRIOR_ALPHA, PRIOR_BETA)
    prior_theta = prior_dist.sample(PRIOR_SAMPLES)
    prior_zeta = logit(prior_theta)
    init_mu = tf.math.reduce_mean(prior_zeta)
    init_omega = tf.math.log(tf.math.reduce_std(prior_zeta))
    mu = tf.Variable(initial_value=init_mu, trainable=True, name='mu')
    omega = tf.Variable(initial_value=init_omega, trainable=True, name='omega')
    print(f'Initial parameters  : mu: {init_mu:1.3f}, omega: {init_omega:1.3f}')
    print(f'Target parameters   : mu: {target_mu:1.3f}, omega: {target_omega:1.3f}')
    print(f'Analytical posterior: alpha: {posterior_alpha}, beta: {posterior_beta}')

    # training the model
    optim = tf.keras.optimizers.RMSprop(learning_rate=5e-2)
    num_epochs = 2000
    for idx in range(num_epochs):
        loss = train_step(mu, omega, prior_dist, data, optim)
        loss = loss.numpy()
        if idx % 100 == 0:
            print(f'[{idx:04d}]: loss: {loss:1.3f} mu: {mu.numpy():1.3f}, omega: {omega.numpy():1.3f}')

    print(f'[{idx:04d}]: loss: {loss:1.3f} mu: {mu.numpy():1.3f}, omega: {omega.numpy():1.3f}')

    print('ho gaya')
