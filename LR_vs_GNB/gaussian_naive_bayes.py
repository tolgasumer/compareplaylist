import numpy as np

class GaussianNaiveBayes: 
    def __init__(self):
        self.classes = dict()
        self.description = dict()

    def gaussian_pdf(self, x, m, std):        
        """Compute the output of the Gaussian probability density function.

        Parameters
        ----------
        x: float
            Input value.
        m: float
            Mean of the distribution.
        std: float
            Standard deviation of the distribution. 

        Returns
        -------
        float
            Probability value.
        """
        return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(- (x - m)**2 / (2 * std**2))

    def fit(self, xtr, ttr): 
        """Fit the model based on training data (training phase).

        Parameters
        ----------
        xtr: array-like
            Training data.
        ttr: array-like
            Target data relative to training features.

        Returns
        -------
            References the instance object. 
        """
        self.n_features = len(xtr[0])
        for i, j in zip(xtr, ttr): ##separates the training instances by class value
            try:
                self.classes[j] += [i]
            except:
                self.classes[j] = [i]
        for value, instances in self.classes.items(): 
            self.description[value] = [[] for i in range(self.n_features)]
            for k in range(self.n_features): 
                h = list(zip(*instances))[k]
                ##store the mean and standard deviation of each attribute for a class value
                self.description[value][k] = (np.mean(h), np.std(h)) 
        return self

    def predict(self, xte):
        """Predict class labels (testing phase).

        Parameters
        ----------
        xte: array-like
            New data to be assigned to classes.

        Returns
        -------
        array-like
            Labels. 
        """
        results = []
        for x in xte: 
            max_prob, max_label = 0.0, None
            for y, s in self.description.items():
                prob = 1.0
                for i in range(self.n_features):
                    prob *= self.gaussian_pdf(x[i], s[i][0], s[i][1]) ##probability that a data instance belongs to a class
                if prob > max_prob: ##select the class with the largest probability
                    max_prob, max_label = prob, y 
            results += [max_label]
        return results