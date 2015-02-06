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

