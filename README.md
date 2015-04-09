kaggle- National Data Science Bowl
==============

Code used for competing in National Data Science Bowl (http://www.kaggle.com/c/datasciencebowl).
The final solution used Convolutional Neural Networks.

Generating the solution
========


Install the dependicies
-----------

Dependices are found in requirements.txt.
I created a even folded where these libraries are stored.


Generating Training and Testing images
------------------------

create ```Data_converted/train/``` by running ```python gen_train.py```

create ```Data_converted/test/``` by running ```python gen_test.py```


Create Final Dataset
-------------------------

Within ```Fish Bowl.ipynb``` run steps 1 & 2 to create the training and testing set needed as inputs to the CNNs.


Pretrain the network
---------------------

We can perform unsupervised pre-training on the network by running the exact network used for training BUT we use regression and the labels are the same as the input features. The pre-training is ran with the full test + train set and the weights of the network are saved into a pickled object. These weights are then used to initialize the true training network.  

```python2.7 run_autoencoder.py fit```

THIS MADE NO IMPROVEMENT SO WILL NOT CONTINUE TO DO IT

Train the network
--------------------

To train the best single mode, run:

```python2.7 run_analysis.py fit```

This will create a pickled object ```net-specialists.pickle``` which contains the neccessary weights to create predictions. 


Generate augmented predictions
----------------------------

To generate predictions which are averaged across multiple transformations of the input, run:

```python2.7 run_predict.py predict```

This will create multiple a csv file with predictions for each test set observation


Single Model predictions
----------------------

To generate predictions for a single model run step 3 within ```Fish Bowl.ipynb```

Blended augmented predictions
---------------------------

To generate predictions for a multiple models averaged together run step 4 within ```Fish Bowl.ipynb```

Train and Predict all models
---------------------

In the end I trained 8 different models, to train and predict all of these at once run ```./run_all.sh``` and ```./predict_all.sh```


Lessons Learned
-------------

Throughout the competition I had repeated issues with data augmentation, I was only able to achieve good results with the rotation of [0,90,180,270]. 
The background in the images is white (255) and opencv/scikit-image assume by default that it is black (0). Therefore, we can invert the images with
```im = np.invert(im)``` when loading in the images. 