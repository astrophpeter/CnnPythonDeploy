# CnnPythonDeploy
Deployment of Cnn in Python Script.


# Contents

1. `model/` contains all the data requited to build the trained tensor
    flow model.
... `model/model-v1.json` meta informationa about model topology used to build the tensorflow graph
... `model/model-v1.h5` binary fule containing all of the train weights and baises which populate the tensorflow graph
... `model/classes.npy` contains the mapping from the human readable classes ('SN1a,'SNII etc..) to machine readble classes ([1,3,0...])

2.  `utils.py` python script containing the prediction function which takes Gaia Spectra and returns classification using tensorflow model. Usage and explaination below.

3. `examples.py` python script showing how to use the predict function.

4. `spec_all.npy' Saved array of some Gaia Spectra - not required only used test as input to predict function.

5. 

# Prerequisites

You will need the follow packages, they can all be installed via pip

1. Tensorflow  [https://www.tensorflow.org/install/]

``` $pip install tensorflow```

2. Keras [https://keras.io/#installation]

``` $pip install keras```

3. H5py [http://docs.h5py.org/en/latest/build.html#install]

``` $pip install h5py```

5. Scikit-learn [http://scikit-learn.org/stable/install.html]

``` $pip install -U scikit-learn```

# Example Usage

This example shows how to classify one GaiaSpectrum which is inputted as a numpy array of length 120
corresponding the number of pixels. The output results contain the predicted class and the corresponding
softmax probabilty associated with that classification. More example usage can by found in the `example.py` script. 

```
>>> GaiaSpectrum = numpy.array([0.234,...,1.344]
>>> result = predict(GaiaSpectrum)
>>> result['class']
['SN1a']
>>> result['prob']
[0.9978]
```
