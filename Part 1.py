from sklearn.neural_network import MLPClassifier

X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
y = [0, 1, 1, 0]

clf = MLPClassifier(activation='logistic', solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(3), random_state=1)

clf.fit(X, y)
print(clf.predict(X))

# Below would give the weights for each neuron
print(clf.coefs_)

# Below would give the intercepts
print(clf.intercepts_)