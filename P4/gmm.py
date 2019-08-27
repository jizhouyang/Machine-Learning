import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None


    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape
        self.gamma = None
        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            self.means, self.gamma, iters = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e).fit(x)
            self.gamma=(self.gamma == np.expand_dims(np.arange(self.n_cluster, dtype=int), axis=1)).astype(float).T
            num = np.sum(self.gamma, axis=0)

            self.variances = np.zeros((self.n_cluster, D, D))

            def vectorize_K(k):
                Temp = x - self.means[k]
                Temp_Matrix = np.multiply(Temp.T, self.gamma[:, k])
                self.variances[k] = np.matmul(Temp_Matrix, Temp) / (num[k]+(1e-10))

            np.vectorize(vectorize_K)(np.arange(self.n_cluster, dtype=int))
            # print (self.variances)
            self.pi_k = num / N
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            
            # Initialize N, mean, var, PI
            Num_cluster=self.n_cluster
            self.pi_k = np.full(Num_cluster, 1 / Num_cluster)
            Temp_var=[np.eye(D)] * Num_cluster
            self.variances = np.array(Temp_var)
            Mean_shape=(Num_cluster, D)
            self.means=np.random.rand(Num_cluster, D)
            # Initialize gamma
            def vectorize_sigma1(k):
                rank = np.linalg.matrix_rank(self.variances[k])
                while rank < D:
                    self.variances[k] += np.eye(D) * 1e-3
                    rank = np.linalg.matrix_rank(self.variances[k])
            np.vectorize(vectorize_sigma1)(np.arange(self.n_cluster,dtype=int))
            temp3=x - np.expand_dims(self.means,axis=1)
            f=np.exp(-0.5 * np.sum(np.multiply(np.matmul(temp3,np.linalg.inv(self.variances)),temp3),axis=2))/(np.sqrt((2 * np.pi) ** D * np.linalg.det(self.variances)).reshape(self.n_cluster,1))
            self.gamma=np.multiply(self.pi_k.reshape(self.n_cluster,1) ,f).T
            self.gamma = (self.gamma.T / np.sum(self.gamma, axis=1)).T
            
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity

        # DONOT MODIFY CODE ABOVE THIS LINE
        l=self.compute_log_likelihood(x)
        for iteration in range(self.max_iter):
            # E step
            self.gamma = (self.gamma.T / np.sum(self.gamma, axis=1)).T
            n = np.sum(self.gamma, axis=0)
            # M step
            self.means=np.matmul(self.gamma.T,x)/(n.reshape(self.n_cluster,1)+(1e-10))
            for k in range(self.n_cluster):self.variances[k] = np.matmul(np.multiply((x - self.means[k]).T, self.gamma[:, k]), x - self.means[k]) / (n[k]+(1e-10))
            #self.variances=np.matmul(np.multiply(np.expand_dims(gamma.T,axis=1),x.T-np.expand_dims(self.means,axis=2)),x-np.expand_dims(self.means,axis=1))/(np.expand_dims(np.expand_dims(n,axis=1),axis=2))
            self.pi_k = n / N
            l_new = self.compute_log_likelihood(x)
            if np.abs(l - l_new) <= self.e:return iteration
            l = l_new

        
        # DONOT MODIFY CODE BELOW THIS LINE

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE

        GMM_samples=np.ones((N, self.means.shape[1]))
        num_clusters=self.n_cluster
        sample_K=np.random.choice(num_clusters, N, p=self.pi_k)
        for index,j in enumerate(sample_K):GMM_samples[index] = np.random.multivariate_normal(self.means[j], self.variances[j])
        return  GMM_samples


        # DONOT MODIFY CODE BELOW THIS LINE

    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        
        N, D = x.shape
        def vectorize_sigma2(k):
            rank = np.linalg.matrix_rank(self.variances[k])
            while rank < D:
                self.variances[k] += np.eye(D) * 1e-3
                rank = np.linalg.matrix_rank(self.variances[k])
        np.vectorize(vectorize_sigma2)(np.arange(self.n_cluster, dtype=int))
        determinant = np.linalg.det(self.variances)
        denominator = np.sqrt((2 * np.pi) ** D * determinant).reshape(self.n_cluster, 1)
        temp3 = x - np.expand_dims(self.means, axis=1)
        f = np.exp(-0.5 * np.sum(np.multiply(np.matmul(temp3, np.linalg.inv(self.variances)), temp3), axis=2)) / denominator
        self.gamma = np.multiply(self.pi_k.reshape(self.n_cluster, 1), f).T
        return float(np.sum(np.log(np.sum(self.gamma, axis=1))))
        # DONOT MODIFY CODE BELOW THIS LINE
    class Gaussian_pdf():
        def __init__(self,mean,variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None: 
            '''
            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to ((2pi)^D) * det(variance) (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit
            # DONOT MODIFY CODE ABOVE THIS LINE
            #raise Exception('Impliment Guassian_pdf __init__')
            # DONOT MODIFY CODE BELOW THIS LINE
            D = variance.shape[0]
            rank = np.linalg.matrix_rank(self.variance)
            while rank < D:
                self.variance += np.eye(D) * 1e-3
                rank = np.linalg.matrix_rank(self.variance)
            self.inv = np.linalg.inv(self.variance)
            self.c = ((2 * np.pi) ^ D) * det(self.variance)

        def getLikelihood(self,x):
            '''
                Input:
                    x: a 1 X D numpy array representing a sample
                Output:
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint:
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)'/sqrt(c))
                    where ' is transpose and * is matrix multiplication
            '''
            #TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE
            #raise Exception('Impliment Guassian_pdf getLikelihood')
            # DONOT MODIFY CODE BELOW THIS LINE
            p = np.exp(-0.5 * np.dot(np.dot((x - self.mean), (self.inv)), (x - self.mean))) / np.sqrt(self.c)
            return p
