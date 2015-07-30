Weighted multiple-instance learning
======================
The attached code implements the multiple-instance learning algorithm for aspect-based 
sentiment analysis which was proposed in the paper listed below. Moreoever, the features 
extracted from seven datasets are provided for research purposes. If you use the code or
features in your research please cite the following paper:

<ul><li>Nikolaos Pappas, Andrei Popescu-Belis, <i>Explaining the Stars: Weighted Multiple-Instance Learning for Aspect-Based Sentiment Analysis</i>, Conference on Empirical Methods in Natural Language Processing (EMNLP), 2014
<br /> <a href="http://publications.idiap.ch/downloads/papers/2014/Pappas_EMNLP14_2014.pdf" target="_blank">http://publications.idiap.ch/downloads/papers/2014/Pappas_EMNLP14_2014.pdf</a>
</li></ul>

Dependencies
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

Running
------------
The code extends BaseEstimator class from scikit-learn package, so you should be able to use it as a common sklearn estimator (check more details on http://scikit-learn.org/stable/). For example:
```bash
$ python
>>> import pickle
>>> from ap_weights import APWeights
>>> from sklearn.metrics import mean_absolute_error
>>> data = pickle.load(open('features/ted_comments.p'))
>>> x = data['X'][:300]
>>> y = data['Y'][:300]
>>> model = APWeights(20, e1=0.5, e2=0.5, e3=0.5)
>>> model.fit(x,y)
[+] Training...
--/start
iteration 0 -> (MAE: 0.063036) 
iteration 1 -> (MAE: 0.059849) 
iteration 2 -> (MAE: 0.059426) 
iteration 3 -> (MAE: 0.059319) 
iteration 4 -> (MAE: 0.059292) 
iteration 5 -> (MAE: 0.059303) 
--/end
>>> mean_absolute_error(model.predict(x),y)
0.067877336959968837
```
