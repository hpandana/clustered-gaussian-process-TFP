import numpy as np
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import norm

import tensorflow_probability as tfp
tfd = tfp.distributions

from .GPR_noiseless import GPR_hmc_noiseless
from .GPR import GPR_hmc


class mixGPR:
  
  def __init__(self, X, y, numK,
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

    assert numK>=2, "numK must be >=2"

    if gpr_observation_noise_variance_prior is None or gpr_observation_noise_variance_initial is None:
      self.isNoiseless = True
    else:
      self.isNoiseless = False

    self.X = X
    self.y = y
    self.numK = numK
    self.num_samples = num_samples
    self.max_em_iter = max_em_iter
    self.gpr_amplitude_prior = gpr_amplitude_prior
    self.gpr_length_scale_prior = gpr_length_scale_prior
    self.gpr_observation_noise_variance_prior = gpr_observation_noise_variance_prior
    self.gpr_amplitude_initial = gpr_amplitude_initial
    self.gpr_length_scale_initial = gpr_length_scale_initial
    self.gpr_observation_noise_variance_initial = gpr_observation_noise_variance_initial
    self.gpr_num_results = gpr_num_results
    self.gpr_num_burnin_steps = gpr_num_burnin_steps
    self.gpr_step_size = gpr_step_size
    self.gpr_num_leapfrog_steps = gpr_num_leapfrog_steps

    kmeans = KMeans(self.numK).fit(self.X)
    self.z = kmeans.labels_
    # check: each class needs to have at least 2 members, else redraw
    _Zclustering_OK = True
    for k in range(self.numK):
      _Zclustering_OK = _Zclustering_OK and (len(np.argwhere(self.z == k)) >= 2)
    assert _Zclustering_OK, "Each class needs to have at least 2 members, please set a different numK"
    
    # Pk holds indices of z belonging to k
    self.Pk = dict()

    # GPR_k holds instance of GPR()
    self.GPR_k= dict()

    #gfun: LDA
    self.gfun = LinearDiscriminantAnalysis()
  
  def _Estep(self):
    # GPR part: sample mean and std, assume normalilty, calculate prob using norm.pdf
    mu = np.array([None]*self.y.shape[0]*self.numK).reshape((self.y.shape[0], self.numK))
    sigma = np.array([None]*self.y.shape[0]*self.numK).reshape((self.y.shape[0], self.numK))
    gpr_prob = np.array([None]*self.y.shape[0]*self.numK).reshape((self.y.shape[0], self.numK))

    pik = np.array([None]*self.y.shape[0]*self.numK).reshape((self.y.shape[0], self.numK))

    for k in range(self.numK):
      samples_k= self.GPR_k[k].predict_samples(self.X, num_samples= self.num_samples)
      mu[:, k] = samples_k.mean(axis=0)
      sigma[:, k] = samples_k.std(axis=0)

    # gfun part:
    gfun_prob= self.gfun.predict_proba(self.X)

    # multiply them together
    for i in range(self.y.shape[0]):
      for k in range(self.numK):
        gpr_prob[i,k] = norm.pdf(self.y[i], loc= mu[i,k], scale= sigma[i,k])
        pik[i,k] = gpr_prob[i,k] * gfun_prob[i,k]
        
    pik = pik / np.sum(pik, axis=1).reshape((self.y.shape[0] ,1))

    # draw z: each class needs to have at least 2 members, else redraw 
    def drawZ():
      for i in range(self.y.shape[0]):
        self.z[i] = np.random.choice(np.array(range(self.numK)), p = np.array(pik[i, :], dtype='float64') )

    drawZ_OK = False
    while not drawZ_OK:
      drawZ()
      _drawZ_OK = True
      for k in range(self.numK):
        _drawZ_OK = _drawZ_OK and (len(np.argwhere(self.z == k)) >= 2)
      drawZ_OK = _drawZ_OK
      
    print ("z= {}".format(self.z))

  def _Mstep(self):
    # populate Pk
    self.Pk = dict()
    for k in range(self.numK):
      ind= np.argwhere(self.z == k)
      self.Pk[k] = ind.reshape((len(ind),))

    # GPR part
    for k in range(self.numK):
      Xk= self.X[self.Pk[k]].reshape((len(self.Pk[k]), self.X.shape[1]))
      yk= self.y[self.Pk[k]]

      if self.isNoiseless:
        self.GPR_k[k]= GPR_hmc_noiseless(Xk, yk,
          self.gpr_amplitude_prior,
          self.gpr_length_scale_prior,
          self.gpr_amplitude_initial,
          self.gpr_length_scale_initial,
          self.gpr_num_results,
          self.gpr_num_burnin_steps,
          self.gpr_step_size,
          self.gpr_num_leapfrog_steps)
      else:
        self.GPR_k[k]= GPR_hmc(Xk, yk,
          self.gpr_amplitude_prior,
          self.gpr_length_scale_prior,
          self.gpr_observation_noise_variance_prior,
          self.gpr_amplitude_initial,
          self.gpr_length_scale_initial,
          self.gpr_observation_noise_variance_initial,
          self.gpr_num_results,
          self.gpr_num_burnin_steps,
          self.gpr_step_size,
          self.gpr_num_leapfrog_steps)

      self.GPR_k[k].fit()

    # gfun part
    self.gfun.fit(self.X, self.z)

  def fit(self):
    em_iter= 0
    isConverged = False
    # print ("z= {}".format(self.z))
    curr_z = self.z.copy()
    
    while not isConverged and em_iter < self.max_em_iter:
      print ('ITER= %s'  %(em_iter) )

      self._Mstep()
      self._Estep()

      _checkConverge= True
      for i in range(self.y.shape[0]):
        _checkConverge = _checkConverge and (curr_z[i] == self.z[i])
      isConverged = _checkConverge

      curr_z = self.z.copy()
      em_iter += 1

  def predict_samples(self, Xnew, num_samples=None):
    assert Xnew.shape[1] == self.X.shape[1], "Xnew and X have to have the same number of features"
    if num_samples is None:
      num_samples = self.num_samples
    pred_samples= np.array([None]*Xnew.shape[0]*num_samples).reshape((num_samples, Xnew.shape[0]))

    gfun_prob= self.gfun.predict_proba(Xnew)
    # print (gfun_prob)
    for i in range(Xnew.shape[0]):
      znew_i = np.random.choice(np.array(range(self.numK)), size= num_samples, p=gfun_prob[i])
      _nk0 = 0
      for k in range(self.numK):
        num_samples_k = np.sum(znew_i == k)
        pred_samples[_nk0:_nk0+num_samples_k, i]= self.GPR_k[k].predict_samples(Xnew[i].reshape((1, Xnew.shape[1])), num_samples_k)
        _nk0 += num_samples_k

    return np.array(pred_samples, dtype=np.float64)
