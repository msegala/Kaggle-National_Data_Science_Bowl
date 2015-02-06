SegCORP-Trader
==============

Machine Learning libraries for quantitative trading.
This package hosts a suit of tools for predicting cryptocurrency prices.


Dependicies
-----------

 * python 2.7
 * Theano 0.6.0
 * pandas 0.13.1
 * sklearn 0.15.0
 * numpy 1.8.1
 * matplotlib 1.3.1
 * scipy 0.14.0
 * IPython 1.2.1
 * tweepy (pip install tweepy)
 * pyalgotrade
 * ws4py
 * tornado
 * nose (pip install nose) for unit testing

Logging into GPU cluster
------------------------

ssh -p 24 segalam@pg62.com

password: RobQuemulg

scp -P 24 mnist.pkl.gz segalam@pg62.com:/home/segalam/Data

nohup sudo THEANO_FLAGS='cuda.root=/opt/cuda,device=gpu,floatX=float32' python2.7 mlp.py > mlp_nohup.out &

ssh -p 24 -L 9000:localhost:8888 segalam@pg62.com (open browser to localhost:9000)

iPython notebooks stored in /var/lib/ipynotebook