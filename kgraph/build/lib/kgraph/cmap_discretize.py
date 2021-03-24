## Import statements
# netcdf/numpy/xray
import numpy as np

# import subplots function for plotting
import matplotlib
from matplotlib import cm

def cmap_discretize(cmap, n_colors=10):
	"""Return discretized colormap.
	from Joe Hamman (https://github.com/jhamman/tonic/blob/master/tonic/plot_utils.py#L66-L94)
	Parameters
	----------
	cmap : str or colormap object
		Colormap to discretize.
	n_colors : int
		Number of discrete colors to divide `cmap` into.
	Returns
	----------
	disc_cmap : LinearSegmentedColormap
		Discretized colormap.
	"""
	import matplotlib
	try:
		cmap = cm.get_cmap(cmap)
	except:
		cmap = cm.get_cmap(eval(cmap))
	colors_i = np.concatenate((np.linspace(0, 1., n_colors), (0., 0., 0., 0.)))
	colors_rgba = cmap(colors_i)
	indices = np.linspace(0, 1., n_colors + 1)
	cdict = {}
	for ki, key in enumerate(('red', 'green', 'blue')):
		cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki])
					  for i in range(n_colors + 1)]

	return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % n_colors,
											  cdict, 1024)
