# Credit_Risk_Analysis
Module 17: Supervised Machine Learning and Credit Risk

# Overview of the Analysis
In this analysis we will be using Machine learning to process a large amount of data to build and evaluate Credit Risk. By using Python and Scikit-learn we will be able to predict credit risk. This will help us establish the strengths and weaknesses of our machine learning models and determine how well the models work. 

# Results
In our evaluation we have used six machine learning models and the results go as follow:

## Oversampling
1.	Random Over Sampler, in this model minority class are randomly selected and added to the training set until the majority and minority classes are balanced. We can observe that the accuracy is 64%
![Random_Forest_Classifier_icr.jpeg](https://github.com/cynmmarin/Credit_Risk_Analysis/blob/0b543e67f6d99f574bc5a831afc1fee918c78c26/Images/Random_Forest_Classifier_icr.jpeg)
2.	SMOTE, is a similar model to the Random Over Sampler, however the way that is balanced is that the minority class a number of its closest neighbors are chosen. We can observed that the accuracy is 66%.
![SMOTE_Oversampling_icr.jpeg](https://github.com/cynmmarin/Credit_Risk_Analysis/blob/0b543e67f6d99f574bc5a831afc1fee918c78c26/Images/SMOTE_Oversampling_icr.jpeg)

## Undersample
1.	ClusterCentroids in this model the algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. We can observed that the accuracy is 54%.
![Undersampling_icr.jpeg](https://github.com/cynmmarin/Credit_Risk_Analysis/blob/0b543e67f6d99f574bc5a831afc1fee918c78c26/Images/Undersampling_icr.jpeg)

## Combination
SMOTEENN model the the minority class is oversampled; however, an undersampling step is added, removing some of each class's outliers from the dataset. The result is that the two classes are separated more cleanly. We can observed that the accuracy is 67%.
![Combination_Sampling_icr.jpeg](https://github.com/cynmmarin/Credit_Risk_Analysis/blob/0b543e67f6d99f574bc5a831afc1fee918c78c26/Images/Combination_Sampling_icr.jpeg)

Next we compared two new machine learning models that reduce bias and predict the credit risk.
1.	Balanced Random Forest Classifier model we can observed that the accuracy is 79%.
![Random_Forest_Classifier_icr.jpeg](https://github.com/cynmmarin/Credit_Risk_Analysis/blob/0b543e67f6d99f574bc5a831afc1fee918c78c26/Images/Random_Forest_Classifier_icr.jpeg)
2.	Easy Ensemble Classifier model we can observed that the accuracy is 93%.
![Easy_Ensemble_AdaBoost_Classifier_icr.jpeg](https://github.com/cynmmarin/Credit_Risk_Analysis/blob/0b543e67f6d99f574bc5a831afc1fee918c78c26/Images/Easy_Ensemble_AdaBoost_Classifier_icr.jpeg)

# Summary

Each machine learning model apply their metrics to best predict the credit risk, however their accuracy is low and can potentially create errors that would damage the ability to determine credit risk. When dealing with such delicate subject is best to have high accuracy of the model, the highest accuracy comes from the Easy Ensemble Classifier model. Given the data we can recommend we can focus on this model but be caution of its inability to fully predict credit risk. 

