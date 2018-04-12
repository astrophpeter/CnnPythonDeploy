######################
# Using a tensorflow #
# to predict classes #
# in python.         #
# @author P. McGill  #
######################

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import time

def predict(GaiaSpectrum,
		modelDir='model/model-v1.',
		classEncoderDir='model/classes.npy'):
	"""
	Predict the class of a Gaia Spectrum and softmax probabilty using a tensflow
	trained model.


	Args:
	   GaiaSpectrum (numpy array): GaiaSpectrum input must either 1 array of dim [120,1]
				       containing 1 Gaia Spectrum or an array of dim [120,n]
				       containing n Gaia Spectra. Elements of array must be 
				       floats.

	   modelDir (String) : Model Directory String

	   classEncoderDir (String) : Class Encoding Directory String

	Return: 
	   result (Dictionary) : Dictionary containing the predicted class result['class']
				 and the corresponding softmax probability result['prob']  

	Example Usage:
		>>> result = predict(numpy.array([0.234,...,1.344]))
		>>> result['class']
		['SN1a']
		>>> result['prob']
		[0.9978]
	"""
	
	# load json and create model
	json_file = open(modelDir + str('json'), 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	# load weights into new model
	loaded_model.load_weights(modelDir + str('h5'))

	# load class encodings
	encoder = LabelEncoder()
	encoder.classes_ = np.load(classEncoderDir)

	# Reshape Spectrum for input to tensorflow model
	if len(GaiaSpectrum.flatten()) == 120:
		GaiaSpectrum = np.array([GaiaSpectrum])
	GaiaSpectrum = GaiaSpectrum.reshape(GaiaSpectrum.shape[0], 1, 120, 1)
	
	#Predict Class
	pred = loaded_model.predict(GaiaSpectrum)
	
	#get the predicted class name and probability
	className = encoder.inverse_transform(np.argmax(pred,axis=1))
	prob = np.max(pred,axis=1)

	return {'class': className.astype(str), 'prob' : prob}

#if __name__ == '__main__':
#	
#	spec = np.load('spec_all.npy')
#	print('Timing classification of ' + str(len(spec)) + ' spectra')
#	start = time.time()
#	result = predict(spec)
#	end = time.time()
#	print('Time taken per Spectrum: ' + str((end-start)/len(spec)*1000) + ' ms')

