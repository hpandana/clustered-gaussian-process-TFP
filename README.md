# Clustered Gaussian Process (GP) Regression in TFP

This is an implementation of clustered GP regression in `tensorflow_probability` (TFP), based on this paper: [A clustered Gaussian process model for computer experiments](https://arxiv.org/abs/1911.04602) by Sung et al (2009).

The GP itself marginalizes hyperparameters with Hamiltonian Monte Carlo integration. It follows the [GPR example](https://github.com/tensorflow/probability/blob/v0.11.0/tensorflow_probability/examples/jupyter_notebooks/Gaussian_Process_Regression_In_TFP.ipynb) provided in `tensorflow_probability` (TFP).
