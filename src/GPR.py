import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()


class GPR_hmc:
  
  def __init__(self, X, y,
    amplitude_prior = tfd.LogNormal(loc=0, scale=np.float64(1.)),
    length_scale_prior = tfd.LogNormal(loc=0, scale=np.float64(1.)),
    observation_noise_variance_prior = tfd.LogNormal(loc=0, scale=np.float64(1.)),
    amplitude_initial = 1.,
    length_scale_initial = 1.,
    observation_noise_variance_initial = 1.,
    num_results = 100,
    num_burnin_steps = 50,
    step_size = 0.1,
    num_leapfrog_steps = 8,
    ):

    assert X.shape[0] == y.shape[0], "X and y have to have the same number of rows."
    assert len(y.shape) == 1, "Cannot have multi dimensional y."

    self.X = X
    self.y = y

    # Create the GP prior distribution
    def build_gp(amplitude, length_scale, observation_noise_variance):
      # Create the covariance kernel
      kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)

      return tfd.GaussianProcess(
        kernel=kernel,
        index_points=self.X,
        observation_noise_variance=observation_noise_variance)

    self.gp_joint_model = tfd.JointDistributionNamed({
      'amplitude': amplitude_prior,
      'length_scale': length_scale_prior,
      'observation_noise_variance': observation_noise_variance_prior,
      'observations': build_gp,
    })

    self.initial_state = [tf.cast(i, tf.float64) for i in [amplitude_initial, length_scale_initial, observation_noise_variance_initial]]

    self.num_results = num_results
    self.num_burnin_steps = num_burnin_steps
    self.step_size = step_size
    self.num_leapfrog_steps = num_leapfrog_steps
    
    self.samples = None

  def fit(self):

    def target_log_prob(amplitude, length_scale, observation_noise_variance):
      return self.gp_joint_model.log_prob({
        'amplitude': amplitude,
        'length_scale': length_scale,
        'observation_noise_variance': observation_noise_variance,
        'observations': self.y
      })

    @tf.function(autograph=False, experimental_compile=False)
    def do_sampling():

      constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Softplus())

      _bijector=[constrain_positive, constrain_positive, constrain_positive]

      sampler = tfp.mcmc.TransformedTransitionKernel(
        tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob,
          step_size=tf.cast(self.step_size, tf.float64),
          num_leapfrog_steps=self.num_leapfrog_steps),
        bijector=_bijector)

      adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=sampler,
        num_adaptation_steps=int(0.8 * self.num_burnin_steps),
        target_accept_prob=tf.cast(0.75, tf.float64))

      return tfp.mcmc.sample_chain(
        kernel=adaptive_sampler,
        current_state=self.initial_state,
        num_results=self.num_results,
        num_burnin_steps=self.num_burnin_steps,
        trace_fn=None)

    self.samples = do_sampling()
  
  def predict_samples(self, Xnew, num_samples):
    assert Xnew.shape[1] == self.X.shape[1], "Xnew and X have to have the same number of features"

    (amplitude_samples, length_scale_samples, observation_noise_variance_samples) = self.samples

    batch_of_posterior_kernels = tfk.ExponentiatedQuadratic(amplitude_samples, length_scale_samples)

    # The batch of kernels creates a batch of GP predictive models, one for each posterior sample.
    batch_gprm = tfd.GaussianProcessRegressionModel(
      kernel=batch_of_posterior_kernels,
      index_points=Xnew,
      observation_index_points=self.X,
      observations=self.y,
      observation_noise_variance=observation_noise_variance_samples,
      predictive_noise_variance=0.)

    # To construct the marginal predictive distribution, we average with uniform
    # weight over the posterior samples.
    predictive_gprm = tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(logits=tf.zeros([self.num_results])),
      components_distribution=batch_gprm)
    
    return predictive_gprm.sample(num_samples).numpy()