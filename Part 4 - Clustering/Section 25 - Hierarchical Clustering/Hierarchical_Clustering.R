#Hierarchical Clustering

#Import the mall dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[, 4:5]

#Using Dendrogram to find number of cluster
dendrogram = hclust(dist(X, method = 'euclidean'),method = 'ward.D')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean Distances')

#Finding Hierarchical Clustering to the mall dataset
hc = hclust(dist(X, method = 'euclidean'),method = 'ward.D')
y_hc = cutree(hc, 5)

#Visualising Hierarchical Clustering results
library(cluster)
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchart = FALSE,
         span = TRUE,
         main = paste('Hierarchical Clustering (Cluster of clients)'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')