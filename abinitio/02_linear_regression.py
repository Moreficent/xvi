"""
Calibrates a linear regression model to the data. The slope and intercept are assumed
to be normally distributed, and the scale of the noise is assumed to follow a half-normal
distribution.

For simplicity, eager mode evaluation is used. 
"""
from abinitio.utils import fix_seed

fix_seed(42)

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

np.set_printoptions(precision=2)

PRIOR_INTERCEPT_LOC = 0
PRIOR_INTERCEPT_SCALE = 1
PRIOR_SLOPE_LOC = 0
PRIOR_SLOPE_SCALE = 1
PRIOR_NOISE_SCALE = 1
PRIOR_SAMPLES = 100
MODEL_SAMPLES = 8
POSTERIOR_SAMPLES = 500


@tf.function
def train_step(mu0, mu1, mu2, omega0, omega1, omega2, prior_intercept_dist, prior_slope_dist, prior_noise_dist, xs, ys,
               optim):
    with tf.GradientTape() as tape:
        eta0 = tf.random.normal([1, MODEL_SAMPLES])
        eta1 = tf.random.normal([1, MODEL_SAMPLES])
        eta2 = tf.random.normal([1, MODEL_SAMPLES])

        theta0 = mu0 + eta0 * tf.math.exp(omega0)
        theta1 = mu1 + eta1 * tf.math.exp(omega1)
        y_mu = theta0 + theta1 * xs

        zeta2 = mu2 + eta2 * tf.math.exp(omega2)
        theta2 = tf.math.exp(zeta2)

        dist = tfp.distributions.Normal(loc=y_mu, scale=theta2)
        lp = dist.log_prob(ys)
        lp = tf.math.reduce_mean(lp, axis=1)
        lp = tf.math.reduce_sum(lp)

        prior_theta0 = prior_intercept_dist.log_prob(theta0)
        prior_theta1 = prior_slope_dist.log_prob(theta1)
        prior_theta2 = prior_noise_dist.log_prob(theta2)
        jac = zeta2
        constraint = tf.reduce_mean(prior_theta0 + prior_theta1 + prior_theta2 + jac) + omega0 + omega1 + omega2

        elbo = lp + constraint
        loss = -elbo

    model_parameters = [mu0, mu1, mu2, omega0, omega1, omega2]
    gradients = tape.gradient(loss, model_parameters)
    optim.apply_gradients(zip(gradients, model_parameters))

    return loss


if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Qt5Agg')

    # generating the data
    data_slope = 0.35
    data_intercept = -0.7
    data_noise_scale = 0.5
    data_samples = 50

    xs = tf.random.uniform(shape=[data_samples, 1], minval=-2, maxval=2)
    noise = tf.random.normal(shape=[data_samples, 1], mean=0, stddev=data_noise_scale)
    ys = data_slope + data_intercept * xs + noise

    # constructing the model
    prior_intercept_dist = tfp.distributions.Normal(PRIOR_INTERCEPT_LOC, PRIOR_INTERCEPT_SCALE)
    prior_intercept_zeta = prior_intercept_dist.sample([PRIOR_SAMPLES])
    init_mu0 = tf.math.reduce_mean(prior_intercept_zeta)
    init_omega0 = tf.math.log(tf.math.reduce_std(prior_intercept_zeta))
    mu0 = tf.Variable(initial_value=init_mu0, trainable=True, name='mu0')
    omega0 = tf.Variable(initial_value=init_omega0, trainable=True, name='omega0')

    prior_slope_dist = tfp.distributions.Normal(PRIOR_SLOPE_LOC, PRIOR_SLOPE_SCALE)
    prior_slope_zeta = prior_slope_dist.sample([PRIOR_SAMPLES])
    init_mu1 = tf.math.reduce_mean(prior_slope_zeta)
    init_omega1 = tf.math.log(tf.math.reduce_std(prior_slope_zeta))
    mu1 = tf.Variable(initial_value=init_mu1, trainable=True, name='mu1')
    omega1 = tf.Variable(initial_value=init_omega1, trainable=True, name='omega1')

    prior_noise_dist = tfp.distributions.HalfNormal(PRIOR_NOISE_SCALE)
    prior_noise_theta = prior_noise_dist.sample([PRIOR_SAMPLES])
    prior_noise_zeta = tf.math.log(prior_noise_theta)
    init_mu2 = tf.math.reduce_mean(prior_noise_zeta)
    init_omega2 = tf.math.log(tf.math.reduce_std(prior_noise_zeta))
    mu2 = tf.Variable(initial_value=init_mu2, trainable=True, name='mu1')
    omega2 = tf.Variable(initial_value=init_omega2, trainable=True, name='omega2')

    print('Initial parameters')
    print(f'  > mu0: {init_mu0:1.3f}, omega0: {init_omega0:1.3f}')
    print(f'  > mu1: {init_mu1:1.3f}, omega1: {init_omega1:1.3f}')
    print(f'  > mu2: {init_mu2:1.3f}, omega2: {init_omega2:1.3f}')

    # training the model
    optim = tf.keras.optimizers.RMSprop(learning_rate=5e-2)
    num_epochs = 2000
    for idx in range(num_epochs):
        loss = train_step(mu0, mu1, mu2, omega0, omega1, omega2, prior_intercept_dist, prior_slope_dist,
                          prior_noise_dist, xs, ys, optim)
        if idx % 100 == 0:
            loss = loss.numpy()
            rep0 = f'mu0: {mu0.numpy():1.3f}, omega0: {omega0.numpy():1.3f}'
            rep1 = f'mu1: {mu1.numpy():1.3f}, omega1: {omega1.numpy():1.3f}'
            rep2 = f'mu2: {mu2.numpy():1.3f}, omega2: {omega2.numpy():1.3f}'
            print(f'[{idx:04d}] loss: {loss:3.3f} {rep0} {rep1} {rep2} ')

    # visualizing
    posterior_intercept_samples = tf.random.normal(shape=[POSTERIOR_SAMPLES],
                                                   mean=mu0.numpy(),
                                                   stddev=tf.math.exp(omega0).numpy()).numpy()
    posterior_slope_samples = tf.random.normal(shape=[POSTERIOR_SAMPLES],
                                               mean=mu1.numpy(),
                                               stddev=tf.math.exp(omega1).numpy()).numpy()
    posterior_noise_samples = tf.exp(
        tf.random.normal(shape=[POSTERIOR_SAMPLES], mean=mu2.numpy(), stddev=tf.math.exp(omega2).numpy())).numpy()

    fig, axs = plt.subplots(1, 3)

    ax = axs[0]
    ax.hist(posterior_intercept_samples)
    ax.set_xlabel('intercept')

    ax = axs[1]
    ax.hist(posterior_slope_samples)
    ax.set_xlabel('slope')

    ax = axs[2]
    ax.hist(posterior_noise_samples)
    ax.set_xlabel('noise')

    plt.show()

    print('ho gaya')
