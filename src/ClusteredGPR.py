import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions


def ClusteredGPR(X, y, numK,
  num_samples = 100,
  max_em_iter = 20,
  gpr_amplitude_prior = tfd.LogNormal(loc=0, scale=np.float64(1.)),
  gpr_length_scale_prior = tfd.LogNormal(loc=0, scale=np.float64(1.)),
  gpr_observation_noise_variance_prior = tfd.LogNormal(loc=0, scale=np.float64(1.)),
  gpr_amplitude_initial = 1.,
  gpr_length_scale_initial = 1.,
  gpr_observation_noise_variance_initial = 1.,
  gpr_num_results = 100,
  gpr_num_burnin_steps = 50,
  gpr_step_size = 0.1,
  gpr_num_leapfrog_steps = 8,
  ):

  if numK == 1:
    if gpr_observation_noise_variance_prior is None or gpr_observation_noise_variance_initial is None:
      from .GPR_noiseless import GPR_hmc_noiseless as GPR
      return GPR(X, y,
        gpr_amplitude_prior,
        gpr_length_scale_prior,
        gpr_amplitude_initial,
        gpr_length_scale_initial,
        gpr_num_results,
        gpr_num_burnin_steps,
        gpr_step_size,
        gpr_num_leapfrog_steps)
    else:
      from .GPR import GPR_hmc as GPR
      return GPR(X, y,
        gpr_amplitude_prior,
        gpr_length_scale_prior,
        gpr_observation_noise_variance_prior,
        gpr_amplitude_initial,
        gpr_length_scale_initial,
        gpr_observation_noise_variance_initial,
        gpr_num_results,
        gpr_num_burnin_steps,
        gpr_step_size,
        gpr_num_leapfrog_steps)
  elif numK > 1: 
    from .GPR_mixture import mixGPR
    return mixGPR(X, y, numK,
      num_samples,
      max_em_iter,
      gpr_amplitude_prior,
      gpr_length_scale_prior,
      gpr_observation_noise_variance_prior,
      gpr_amplitude_initial,
      gpr_length_scale_initial,
      gpr_observation_noise_variance_initial,
      gpr_num_results,
      gpr_num_burnin_steps,
      gpr_step_size,
      gpr_num_leapfrog_steps)
