##########################
# A script to demostrate #
# the predict function in#
# untils                 #
# @author P McGill       #
##########################

from utils import predict
import numpy as np

#load the spectra as numpy arrays
spec = np.load('spec_all.npy')

#classification of one spectrum
result = predict(spec[0])
print('Class: ' + str(result['class']))
print('Probability: ' + str(result['prob']))

#classification of 5 spectra
result = predict(spec[550:555])
print('Classes: ' + str(result['class']))
print('Probabilities:  ' + str(result['prob'])) 
