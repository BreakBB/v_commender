from sklearn.naive_bayes import GaussianNB


class NaiveBayesClassifier:
    clf = None

    def __init__(self):
        self.clf = GaussianNB()

    def train(self, training_data, training_votes):
        self.clf.fit(training_data, training_votes)

    def predict(self, movie_vector):
        return self.clf.predict([movie_vector]).tolist()
