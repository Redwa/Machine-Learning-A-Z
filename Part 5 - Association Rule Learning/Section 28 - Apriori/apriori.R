#Apriori

#data preprocessing
#install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',',rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset , topN = 100)


#Training Apriori to dataset
#support = people buy a products 4 times * 7 day(from data) / 7500 transactions
rules = apriori(data=dataset, parameter = list(support = 0.004, confidence = 0.2))

#Visualising the Results
inspect(sort(rules, by = 'lift')[1:10])