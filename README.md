<b>wmil</b> — The attached code is a Python implementation of the multiple-instance learning algorithm for aspect-based 
sentiment analysis which was proposed in the paper listed below. Moreoever, the features 
extracted from seven datasets are provided for research purposes. 

```
@InProceedings{pappas14,
  author    = {Pappas, Nikolaos  and  Popescu-Belis, Andrei},
  title     = {Explaining the Stars: Weighted Multiple-Instance Learning for Aspect-Based Sentiment Analysis},
  booktitle = {Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  month     = {October},
  year      = {2014},
  address   = {Doha, Qatar},
  publisher = {Association for Computational Linguistics},
  pages     = {455--466},
  url       = {http://www.aclweb.org/anthology/D14-1052}
}
```


A more scalable version of the above algorithm based on stochastic gradient descent can be found here: <a href="http://github.com/nik0spapp/wmil-sgd"> wmil-sgd</a>. 

Installing dependencies
------------
The available code requires Python programming language and pip package manager to run. 
For detailed instructions on how to install it along with a package manager please refer 
to the following links: http://www.python.org/getit/ and http://www.pip-installer.org/en/latest/.

Next, you should be able to install the following packages: <br />
```bash
$ pip install numpy 
$ pip install scikit-learn
$ pip install scipy
```

Training and testing the model
------------
The code extends BaseEstimator class from scikit-learn package, so you should be able to use it as a common sklearn estimator (check more details on http://scikit-learn.org/stable/). For example:
```bash
$ python
>>> import pickle
>>> from wmil import APWeights
>>> from sklearn.metrics import mean_absolute_error
>>> data = pickle.load(open('features/ted_comments.p'))
>>> size = len(data['X'])
>>> k = int(size*0.5)
>>> x_train = data['X'][:k]
>>> y_train = data['Y'][:k]
>>> x_test = data['X'][k:]
>>> y_test = data['Y'][k:]
>>> model = APWeights(20, e1=1.0, e2=1.0, e3=1.0)
>>> model.fit(x_train, y_train)
[+] Training...
--/start
iteration 0 -> (MAE: 0.103437)
iteration 1 -> (MAE: 0.089629)
iteration 2 -> (MAE: 0.087793)
iteration 3 -> (MAE: 0.087565)
iteration 4 -> (MAE: 0.087523)
iteration 5 -> (MAE: 0.087515)
iteration 6 -> (MAE: 0.087510)
iteration 7 -> (MAE: 0.087511)
--/end
>>> mean_absolute_error(model.predict(x_train),y_train) # training error
0.096217463769192518
>>> mean_absolute_error(model.predict(x_test), y_test) # testing error
0.16325402985689552
```
