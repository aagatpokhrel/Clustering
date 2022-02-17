import numpy as np

class KMeans:
    def __init__(self, n_clusters = 3, tolerance = 0.01, max_iter = 100, runs = 1):
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.cluster_means = np.zeros(n_clusters)
        self.max_iter = max_iter
        self.runs = runs
        
    def fit(self, X):
        row_count, col_count = X.shape
        X_values = self.get_values(X)
        X_labels = np.zeros(row_count)
        costs = np.zeros(self.runs)
        all_clusterings = []

        for i in range(self.runs):
            cluster_means =  X [ np.random.choice(row_count, size=self.n_clusters, replace=False) ]
            for _ in range(self.max_iter):            
                previous_means = np.copy(cluster_means)
                distances = self.compute_distances(X_values, cluster_means, row_count)
                X_labels = self.label_examples(distances)
                cluster_means = self.compute_means(X_values, X_labels, col_count)
                clusters_not_changed = np.abs(cluster_means - previous_means) < self.tolerance
                if np.all(clusters_not_changed) != False:
                    break
            
            X_values_with_labels = np.append(X_values, X_labels[:, np.newaxis], axis = 1) 
            all_clusterings.append( (cluster_means, X_values_with_labels) )
            costs[i] = self.compute_cost(X_values, X_labels, cluster_means)
        
        best_clustering_index = costs.argmin()
        self.cost_ = costs[best_clustering_index]
        return all_clusterings[best_clustering_index]
        
    def compute_distances(self, X, cluster_means, row_count):
        distances = np.zeros((row_count, self.n_clusters))
        for cluster_mean_index, cluster_mean in enumerate(cluster_means):
            distances[:, cluster_mean_index] = np.linalg.norm(X - cluster_mean, axis = 1)
            
        return distances
    
    def label_examples(self, distances):
        return distances.argmin(axis = 1)
    
    def compute_means(self, X, labels, col_count):
        cluster_means = np.zeros((self.n_clusters, col_count))
        for cluster_mean_index, _ in enumerate(cluster_means):
            cluster_elements = X [ labels == cluster_mean_index ]
            if len(cluster_elements):
                cluster_means[cluster_mean_index, :] = cluster_elements.mean(axis = 0)
                
        return cluster_means
    
    def compute_cost(self, X, labels, cluster_means):
        cost = 0
        for cluster_mean_index, cluster_mean in enumerate(cluster_means):
            cluster_elements = X [ labels == cluster_mean_index ]
            cost += np.linalg.norm(cluster_elements - cluster_mean, axis = 1).sum()
        
        return cost
            
    def get_values(self, X):
        if isinstance(X, np.ndarray):
            return X
        return np.array(X)