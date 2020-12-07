# Random_forest/Company_sales_prediction

A cloth manufacturing company is interested to know about the segment or attributes causes high sale
Have used Random_Forest_Classifier algorithm for prediction of sales.

## Random_Forest
It is an ensemble algorithm or we can call it as bagging technique.Internally "RF"(Random_Forest) is made up of many "DT"(Decision_Tress).One important thing to note here is most of the time DT will overfit but RF dosen't usually.This is because with each and every tree row and column sampling will take place by which tree won't grow till depth. And decision won't be considered based on one model instead from many models for classification. Whichever class is having highest vote that class is selected for that row.When it cames to "regression" average of predicted values will be taken for that row.

## Advantages
1)Does not overfit
2)No need for checking for outliers only if we are using this algo.

## Disadvantages
1)It will be slow if we are using many trees.

## Paramters to be taken care to reduce_overfitting.
1)n_estimators
2)max_depth
3)min_sample_leaf.
