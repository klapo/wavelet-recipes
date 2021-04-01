def SRON(numcats):
# Gives colors for printing figures that have been 
# optimized as defined by the SRON technical note.
# (http://www.sron.nl/~pault/colourschemes.pdf)
#
# SYNTAX:
#	colsche = SRON(numcats)
#
# INPUTS:
#	numcats		= 1x1 scalar, number of categories to plot with distinct
#				colors
#
# OUTPUTS:
#	colsche		= numcats x 3 matrix, RGB color values 

	## Import Libraries
	import numpy as np

	############
	## CHECKS ##
	############
	if numcats > 12:
		raise ValueError('The number of categories must be an integer of 12 or less')

	##########
	## CODE ##
	##########
	# Determine the color scheme depending on the number of categories.
	if numcats == 1:
		colsche = np.array([68.,119.,170.])[np.newaxis,:]
	elif numcats == 2:
		colsche = np.array([[68.,119.,170.],\
					[221,204,119]])
	elif numcats == 3:
		colsche = np.array([[68.,119.,170.],\
					[221.,204.,119],\
					[204.,102.,119.]])
	elif numcats == 4:
		colsche = np.array([[68.,119.,170.],\
					[17.,119.,71.],\
					[221.,204.,119.],\
					[204.,102.,119.]])
	elif numcats == 5:
		colsche = np.array([[51.,34.,138.],\
					[136.,204.,238.],\
					[17.,119.,71.],\
					[221.,204.,119.],\
					[204.,102.,119.]])
	elif numcats == 6:
		colsche = np.array([[51.,34.,138.],\
					[136.,204.,238.],\
					[17.,119.,71.],\
					[221.,204.,119.],\
					[204.,102.,119.],\
					[170.,68.,153.]])
	elif numcats == 7:
		colsche = np.array([[51.,34.,138.],\
					[136.,204.,238.],\
					[68.,170.,153.],\
					[17.,119.,71.],\
					[221.,204.,119.],\
					[204.,102.,119.],\
					[170.,68.,153.]])
	elif numcats == 8:
		colsche = np.array([[51.,34.,138.],\
					[136.,204.,238.],\
					[68.,170.,153.],\
					[17.,119.,71.],\
					[153.,153.,51.],\
					[221.,204.,119.],\
					[204.,102.,119.],\
					[170.,68.,153.]])
	elif numcats == 9:
		colsche = np.array([[51.,34.,138.],\
					[136.,204.,238.],\
					[68.,170.,153.],\
					[17.,119.,71.],\
					[153.,153.,51.],\
					[221.,204.,119.],\
					[204.,102.,119.],\
					[136.,34.,85.],\
					[170.,68.,153.]])
	elif numcats == 10:
		colsche = np.array([[51.,34.,138.],\
					[136.,204.,238.],\
					[68.,170.,153.],\
					[17.,119.,71.],\
					[153.,153.,51.],\
					[221.,204.,119.],\
					[102.,17.,0.],\
					[204.,102.,119.],\
					[136.,34.,85.],\
					[170.,68.,153.]])
	elif numcats == 11:
		colsche = np.array([[51.,34.,138.],\
					[102.,153.,204.],\
					[136.,204.,238.],\
					[68.,170.,153.],\
					[17.,119.,71.],\
					[153.,153.,51.],\
					[221.,204.,119.],\
					[102.,17.,0.],\
					[204.,102.,119.],\
					[136.,34.,85.],\
					[170.,68.,153.]])
	elif numcats == 12:
		colsche = np.array([[51.,34.,138.],\
					[102.,153.,204.],\
					[136.,204.,238.],\
					[68.,170.,153.],\
					[17.,119.,71.],\
					[153.,153.,51.],\
					[221.,204.,119.],\
					[102.,17.,0.],\
					[204.,102.,119.],\
					[170.,68.,102.],\
					[136.,34.,85.],\
					[170.,68.,153.]])

	colsche = np.divide(colsche,256.)

	return colsche
