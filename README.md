# IDS_SSPAE_SVM
This is a code for training and testing a NIDS using a novel supervised auto-encoder named supervised sparse auto-encoder.
the arguments of the code are:

* -option: getting two values pre_train and test. The pre_train is the SSPAE Training process that produces the z_train and z_test. The test train SVM and print the accuracy of testing data on the model. This argument is required.
* -mi: producing mutual information needs much ram (16 GB at least). You could use this option to give the path of the mi file we include in the project root or your produced mi file.
  
* -output giving two values separated by space for transform train and test data (named z_tr and z_te). If the option is pre_train, this argument is required.
* -z_tr: the z_tr path produced by pre_training. This argument is required if the option is testing.
* -z_te: the z_te path produced by pre_training. This argument is required if the option is testing.

##  Training

```
python3 binary_classification.py -option pre_train -mi mi.csv -output z_tr.csv z_te.csv
```
## Testing 
```
 python3 binary_classification.py -option test -z_tr z_train.csv -z_te z_test.csv
```