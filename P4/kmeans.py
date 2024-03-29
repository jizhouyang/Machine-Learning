import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a size (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates an Int)
            Note: Number of iterations is the number of time you update the assignment
        ''' 
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # inialization
        random_index = np.random.choice(N, self.n_cluster, replace=True)
        random_mean = x[random_index]
        J = 1e-10
        R = np.zeros(N).astype(int)

        for iter in range(self.max_iter):
            R = np.argmin(np.sum(((x - np.expand_dims(random_mean, axis=1)) ** 2), axis=2), axis=0)
            # computer distortion
            J_new = np.mean(np.sum((x - random_mean[R]) ** 2, axis=1))
            if np.absolute(J - J_new) <= self.e:
                break
            J = J_new
            # calculate mean
            temp=[]
            for cluster in range(self.n_cluster):temp+=[np.mean(x[R == cluster], axis=0)]
            temp=np.array(temp)
            temp[np.where(np.isnan(temp))] = random_mean[np.where(np.isnan(temp))]
            random_mean = temp
        return (random_mean, R, iter)
        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
            #'Implement fit function in KMeans class (filename: kmeans.py)')
        # DONOT CHANGE CODE BELOW THIS LINE

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float) 
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by 
                    majority voting ((N,) numpy array) 
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels
        centroids, R, iters=KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e).fit(x)
        clustersArr=np.arange(self.n_cluster,dtype=int)
        M_index = (R == np.expand_dims(clustersArr, axis=1)).astype(int)
        def classifer(a):
            temp_index=np.where(a == 1)
            if all(a==0):return 0
            else:return np.argmax(np.bincount(y[temp_index]))
        centroid_labels=np.apply_along_axis(classifer,arr=M_index,axis=1)
        #print(centroid_labels)
        # DONOT CHANGE CODE ABOVE THIS LINE
       # raise Exception(
            #'Implement fit function in KMeansClassifier class (filename: kmeans.py)')

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels
        labels=self.centroid_labels[np.argmin(np.sum(((x - np.expand_dims(self.centroids, axis=1)) ** 2), axis=2), axis=0)]
        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
            #'Implement predict function in KMeansClassifier class (filename: kmeans.py)')
        # DONOT CHANGE CODE BELOW THIS LINE
        return labels

