# opinion-dynamics-code
This repository contains the Python code for the paper "Opinion dynamics on non-sparse networks with community structure" (https://arxiv.org/abs/2401.04598).

## 📂 Files

- `Opinions_Distributions.py`:
  Simulates the opinion process for various density regimes, as well as the mean-field process, and plots their distributions. The approximation is really tight in the dense regimes, while it deteriorates in the sparse regime, as the theory predicts.

- `Opinions_MaxDifferences.py`:  
  Simulates the opinion process for various density regimes and media signals, and computes the maximum differences from the mean-field process. We verify that the denser the underlying network, the more accurate the mean-field approximation.
