#K-Means

#Import Dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

#Using Elbow medthod to Optimal number of cluster
set.seed(6)
wcss = vector()
for (i in 1: 10) wcss[i] = sum(kmeans(X, i) $withinss)
plot(1:10, 
     wcss, 
     type = 'b', 
     main = paste('Cluster of client'), 
     xlab = 'Number of cluster',
     ylab = 'WCSS')

#Apply K-Means to the mall dataset
set.seed(29)
kmeans = kmeans(X, 5,iter.max = 300, nstart = 10)

#Visualising the clusters
library(cluster)
clusplot(X,
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchart = FALSE,
         span = TRUE,
         main = paste('K-Mean (Cluster of clients)'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')