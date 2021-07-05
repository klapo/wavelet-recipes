# wavelet-recipes

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/klapo/wavelet-recipes/HEAD?filepath=notebooks)

Cookbooks for continuous wavelet and biwavelet analysis based on the pycwt libraries. Launch the interactive notebooks through the binder link above. Note that the large size of data used will usually crash the binder notebooks towards the end of the notebooks without manually clearing variables. Monte-Carlo simulations in the biwavelet-significance notebook will run slowly as well.

## Cookbooks

[wavelet-significance](https://nbviewer.jupyter.org/github/klapo/wavelet-recipes/blob/main/notebooks/wavelet-significance.ipynb): How to do significance testing for various types of noise and the importance of providing _a priori_ noise estimates. Includes basic usage examples of the Continuous Wavelet Transform including the scale rectified power spectra.

[cwt_biwavelet_usage](https://nbviewer.jupyter.org/github/klapo/wavelet-recipes/blob/main/notebooks/cwt_biwavelet_usage.ipynb): How to use the biwavelet linear coherence, the cross-wavelet transform, and the cross-wavelet transform's phase relationships. Includes examples comparing to published figures for the scaled power spectra.

[biwavelet-significance](https://nbviewer.jupyter.org/github/klapo/wavelet-recipes/blob/main/notebooks/biwavelet-significance.ipynb): How to do and the development of the Monte-Carlo significance testing code used by the helper functions for biwavelet linear coherence.

# Example applications

An example of the application of these recipes for detecting Internal Gravity Waves in Distributed Temperature Data is shown in my [EGU 2021 presentation](presentations/wavelet-application-example_IGW-EGU-2021.pdf).
