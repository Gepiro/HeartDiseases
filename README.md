# Heart Diseases

Predicting Heart Diseases via Decision Trees & Random Forests with data set from Kaggle: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease .

In order to run:
spark-submit --class HeartDiseases HeartDiseases.jar

# Explanation Work

We splitted the dataframe in two different array with the 90% of the data in the first one and the 10% in the second one:
val Array(trainData, testData) = dataFrame.randomSplit(Array(0.9, 0.1))

We used the StringIndexer for converting the "HeartDisease" label in an integer label and OneHotEncoder for converting also the features.
We merge together the features in an vector with VectorAssembler.
We created the Decision Tree and we passed the whole set of those stages to a Pipeline.
After the creation of a ParamGridBuilder with the params, we created the CrossValidator with 5-folds. 
We calculeted the average metric for each of the cross-validation and hyper-parameter iterations with avgMetrics.
Eventually, we saved the tree.
We obtained the predictionAndLabels for computing precision, recall and accuracy. The last one computed as:
(TP+TN)/Number of example

This part needs less than 30 minutes to finish.
The best tree is the one with maxBins = 20, impurity = "gini" and maxDepth = 5.

We repeted the cross validation with a randomForest.
This part needs about an hour and half.
The best forest is the one with maxBins = 100, impurity = "gini", numTrees = 20 and maxDepth = 15.

At the end, we recreated the model in the original script (starting from a dataframe, so we converted it in RDD[LabelPoint]).
For the original model the best tree is impurity = "gini", maxDepth = 30 and maxBins = 300.
This last part, from the original script, needs only 20 minutes to finish.

Now, we can compare the three parts:
- The tree from the original script needs 20 minutes to finish. The parameters are impurity = "gini", maxDepth = 30 and maxBins = 300. The final accuracy is 0.866504854368932.
- The tree from the first CrossValidation needs less than 30 minutes to finish. The parameters are maxBins = 20, impurity = "gini" and maxDepth = 5. The final accuracy is 0.9140502600739487.
- The forest needs about one hour and half. The parameters are maxBins = 100, impurity = "gini", numTrees = 20 and maxDepth = 15. The final accuracy is Accuracy: 0.9134996721106704.

We can say that, in this situation, those last two parts have a better accuracy rather than the one from the original script, but they need more time.