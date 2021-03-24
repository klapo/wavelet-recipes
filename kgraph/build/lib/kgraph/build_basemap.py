import numpy as np
from mpl_toolkits.basemap import Basemap
import sys, pickle, os

def build_basemap(lon,lat,dir_bmap,bmap_name='basemap.pickle',rewrite=False):
# Build basemaps and save as a pickle file for easy and efficient loading later.
#Parameters
# ----------
# lon 		: numpy array of map longitude
# lat 		: numpy array of map latitude
# dir_bmap 	: string, directory to save pickeled basemap file
# dmap_name	: string, name of output basemap file to either write or read in
# rewrite		: logical True/False, rewrite the pickled basemap file
# 
# Returns
# -------
# bmap 		: basemap object
# bmap_dict	: dictionary, basemap attributes
# 
# Notes
# -----
# 
# See Also
# --------
	# Lat/Lon handling - map extent
	bmap_dict = {}
	bmap_dict['lat_i'] = np.min(lat)
	bmap_dict['lon_i'] = np.min(lon)
	bmap_dict['lat_j'] = np.max(lat)
	bmap_dict['lon_j'] = np.max(lon)
	
	bmap_dict['lat_mid'] = lat[np.round(lat.size/2)]
	bmap_dict['lon_mid'] = lon[np.round(lon.size/2)]
	
	# lat/lon labels
	bmap_dict['lat_labels'] = np.arange(np.round(bmap_dict['lat_i']), np.round(bmap_dict['lat_j']), 2)
	bmap_dict['lon_labels'] = np.arange(np.round(bmap_dict['lon_i']), np.round(bmap_dict['lon_j']), 2)

   	# Change to basemap directory
	os.chdir(dir_bmap)

	# Force rewriting basemap pickle file
	if rewrite:
		bmap = Basemap(llcrnrlon=bmap_dict['lon_i'],llcrnrlat=bmap_dict['lat_i'],\
						urcrnrlon=bmap_dict['lon_j'],urcrnrlat=bmap_dict['lat_j'],\
						rsphere=(6378137.00,6356752.3142),resolution='l',area_thresh=1000.,projection='lcc',\
						lat_1=bmap_dict['lat_mid'],lon_0=bmap_dict['lon_mid'])
		pickle.dump(bmap,open(bmap_name,'wb'),-1)
	
	# Try loading a pre-built basemap
	else:
		try:
			bmap = pickle.load(open(bmap_name,'rb'))
		except IOError as e:
			bmap = Basemap(llcrnrlon=bmap_dict['lon_i'],llcrnrlat=bmap_dict['lat_i'],\
							urcrnrlon=bmap_dict['lon_j'],urcrnrlat=bmap_dict['lat_j'],\
							rsphere=(6378137.00,6356752.3142),resolution='l',area_thresh=1000.,projection='lcc',\
							lat_1=bmap_dict['lat_mid'],lon_0=bmap_dict['lon_mid'])
			pickle.dump(bmap,open(bmap_name,'wb'),-1)
	
	return bmap,bmap_dict
