import numpy as np 

class LogisticRegression:
    def __init__(self, iterations=15000, learning_rate=0.10):
        """
        Parameters
        ----------
        iterations: int
            Maximum number of iterations.
        learning_rate: float
            Controls how much the weights are adjusted given the loss gradient. 
        """
        self.iterations = iterations
        self.learning_rate = learning_rate

    def logistic_function(self, x):
        """Compute the output of the logistic function.

        Parameters
        ----------
        x: float
            Input value.

        Returns
        -------
        float
            A value between 0 and 1.
        """
        return 1 / (1 + np.exp(-x))
    
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
        xtr = np.concatenate((np.ones((xtr.shape[0], 1)), xtr), axis=1) ##add intercept
        self.weights = np.zeros(xtr.shape[1]) ##initialize weights        
        func = self.logistic_function(np.dot(xtr, self.weights))
        for i in range(self.iterations):
            self.weights -= (np.dot(xtr.T, (func - ttr)) / ttr.size) * self.learning_rate
            func = self.logistic_function(np.dot(xtr, self.weights))
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
        xte = np.concatenate((np.ones((xte.shape[0], 1)), xte), axis=1)
        return self.logistic_function(np.dot(xte, self.weights)).round()