# wavelet-recipes

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/klapo/wavelet-recipes/HEAD?filepath=notebooks)

Cookbooks for continuous wavelet and biwavelet analysis based on the pycwt libraries. Launch the interactive notebooks through the binder link above. Note that the large size of data used will usually crash the binder notebooks towards the end of the notebooks without manually clearing variables. Monte-Carlo simulations in the biwavelet-significance notebook will run slowly as well.

## Cookbooks

[wavelet-significance](https://nbviewer.jupyter.org/github/klapo/wavelet-recipes/blob/main/notebooks/wavelet-significance.ipynb): How to do significance testing for various types of noise and the importance of providing _a priori_ noise estimates. Includes basic usage examples of the Continuous Wavelet Transform including the scale rectified power spectra.

[cwt_biwavelet_usage](https://nbviewer.jupyter.org/github/klapo/wavelet-recipes/blob/main/notebooks/cwt_biwavelet_usage.ipynb): How to use the biwavelet linear coherence, the cross-wavelet transform, and the cross-wavelet transform's phase relationships. Includes examples comparing to published figures for the scaled power spectra.

[biwavelet-significance](https://nbviewer.jupyter.org/github/klapo/wavelet-recipes/blob/main/notebooks/biwavelet-significance.ipynb): How to do and the development of the Monte-Carlo significance testing code used by the helper functions for biwavelet linear coherence.

[wavelet-normalization.ipynb](https://nbviewer.jupyter.org/github/klapo/wavelet-recipes/blob/main/notebooks/wavelet-normalization.ipynb): An exploration of the normalization keywords and their impact on the spectral scaling of the CWT spectra. Comparisons are made to an FFT spectra. Necessary when comparing between processes with different underlying variances (e.g. different instruments).

# Example applications

An example of the application of these recipes for detecting Internal Gravity Waves in Distributed Temperature Data is shown in my [EGU 2021 presentation](presentations/wavelet-application-example_IGW-EGU-2021.pdf).

# References
Each reference is referred to by the initial of the first author and year (e.g., Ge (2008) will be G08).

Allen, M. R., and L. A. Smith (1996), Monte Carlo SSA: Detecting Irregular Oscillations in the Presence of Colored Noise, J. Clim., 9, 3373–3403, doi:10.1175/1520-0442(1996)009<3373:MCSDIO>2.0.CO;2.

Chavez, M., and B. Cazelles (2019), Detecting dynamic spatial correlation patterns with generalized wavelet coherence and non-stationary surrogate data, Sci. Rep., 9(April), 1–9, doi:10.1038/s41598-019-43571-2.

Ge, Z. (2007), Significance tests for the wavelet power and the wavelet power spectrum, Ann. Geophys., 25(11), 2259–2269, doi:10.5194/angeo-25-2259-2007.

Ge, Z. (2008), Significance tests for the wavelet cross spectrum and wavelet linear coherence, Ann. Geophys., 26(2007), 3819–3829, doi:10.5194/angeo-26-3819-2008.

Gilman, D. L., F. J. Fuglister, and J. J. M. Mitchell (1962), On the Power Spectrum of “Red Noise,” J. Atmos. Sci., 20, 182–184, doi:1520-0469(1963)020<0182:OTPSON>2.0.CO;2.

Liu, Y., S. X. Liang, and R. H. Weisberg (2007), Rectification of the Bias in the Wavelet Power Spectrum, J. Atmos. Ocean. Technol., 24, 2093–2102, doi:10.1175/2007JTECHO511.1.

Torrence, C., and G. P. Compo (1998), A Practical Guide to Wavelet Analysis, Bull. Am. Meteorol. Soc., 79(1), 61–78, doi:https://doi.org/10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2.

Vamos, C., Soltuz, S. M., & Craciun, M. (2007). Order 1 autoregressive process of finite length. ArXiv.

Veleda, D., R. Montagne, and M. Araujo (2012), Cross-Wavelet Bias Corrected by Normalizing Scales, J. Atmos. Ocean. Technol., 29, 1401–1408, doi:10.1175/JTECH-D-11-00140.1.

Zhang, Z., & Moore, J. C. (2012). Comment on “significance tests for the wavelet power and the wavelet power spectrum” by Ge (2007). Annales Geophysicae, 30(12), 1743–1750. https://doi.org/10.5194/angeo-30-1743-2012