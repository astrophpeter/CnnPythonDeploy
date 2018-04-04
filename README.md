# CnnPythonDeploy
Deployment of Cnn in Python Script


# Contents

1. 'model/' contains all the data requited to build the trained tensor
    flow model.


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
