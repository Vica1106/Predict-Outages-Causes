# Predict-Outages-Causes
Final project for DSC 80 at UCSD

**Name(s)**: Haoyang Yu, Jessie Zhang

### Introduction
Numerous factors contribute to outages, some of which can significantly impact individuals and society as a whole. Among these, severe weather often stands as a leading cause. Our project focuses on forecasting whether a major power outage stems from severe weather conditions. If successful, this initiative aims to streamline the identification of outage triggers, consequently saving valuable time in determining causation.

Moreover, this predictive capability holds promise in aiding environmental agencies. It allows for timely notifications regarding the occurrence of severe weather, prompting proactive measures to mitigate power disruptions stemming from such events. Additionally, by alerting the populace to conserve power during these periods, we aim to maintain a more resilient power grid, reducing the likelihood of future outages precipitated by severe weather conditions.

To make accurate predictions, we will utilize historical data on power outages that have occurred during January 2000 to July 2016. We mainly focuse on the 'MONTH','ANOMALY.LEVEL', 'CAUSE.CATEGORY', 'OUTAGE.DURATION', 'CUSTOMERS.AFFECTED' columns in our dataset.

We plan to train a binery classifier for the prediction model. By training a classification model, we want to develop a prediction to identify whether a power outage is caused by the several weather based on the information we can get after a major power outage.

**Step 0.1**: Only keep columns we want.
**Step 0.2**: Data Cleaning
1. Clean the missing data from column OUTAGE.DURATION and 'CUSTOMERS.AFFECTED' by drop the null values.
2. Change data type of column 'OUTAGE.DURATION' to float.

## Framing the Problem
Our dataset encompasses records of significant power outages observed across various states in the continental U.S. during January 2000–July 2016, comprising 1534 entries and 55 variables. To enhance the analytical focus on power outage research, a decision has been made to streamline the dataset by retaining only 6 pertinent columns. 

**Our dataframe**:


**Our prediction question**: Predict if a major power outage is caused by the severe weather.

**Our prediction type**: We are performing binary classification, which only predict the a major power outage is caused by the severe weather or not.

**Response variables**:
1. 'ANOMALY.LEVEL' serves as an indicator of weather conditions, with lower scores denoting colder weather and higher scores signifying warmer conditions. Our hypothesis suggests that extreme cold or hot weather may induce more severe conditions than typical weather patterns.

2. The 'MONTH' column aligns with a similar rationale to 'ANOMALY.LEVEL', as winter and summer months potentially contribute to increased probabilities of severe weather occurrences.

3. 'OUTAGE.DURATION' denotes the duration of each power outage. Our prediction posits that if a power outage is triggered by severe weather conditions, the outage duration is likely to be prolonged. This extension in time may result from challenges faced by workers in repairing infrastructure during severe weather, necessitating waiting periods for improved weather conditions, thereby elongating the outage duration.

4. 'CUSTOMERS.AFFECTED' represents the number of individuals impacted by the outage. Our prediction indicates that power outages caused by severe weather are likely to affect a larger number of individuals. Severe weather has the potential to affect expansive areas, leading to widespread power outages and consequently impacting a greater number of people within the affected regions.

The data mentioned becomes available post-outage, where the cause is initially unknown. We conduct an analysis subsequent to the outage occurrence, obtaining details such as the anomaly level, the specific month, outage duration, and the count of affected individuals. Leveraging this post-outage information, we aim to employ predictive methodologies to discern and forecast the potential factors that led to the outage.

**Metric using to evaluate**:

We decide to use F1-score to evaluate our model. Since F1-score and accuracy are overall measures of a binary classifier's performance. But we learn in the lecture that accuracy is misleading in the presence of class imbalance, and doesn't take into account the kinds of errors the classifier makes. In order to have a better determination of the model, we decide to use F1-score.

## Baseline Model
**Step 2.1**: Basic category transform
Since we choose to analysis whether the outage is cause by the severe weather, we need to encode the Column 'CAUSE.CATEGORY' to individual cause as a single column with 1 for true and 0 for false. In this way, we will have a column to determine the cause of severe weather. 

**Step 2.2**: Model and features
Model:
1. The chosen model for our analysis is the DecisionTreeClassifier. Its adaptability in capturing intricate relationships and non-linear patterns within the data renders it apt for discerning complexities that linear models might struggle to capture.

2. An advantageous facet of Decision trees lies in their capability to handle missing values within features seamlessly. The inherent structure of the tree accommodates missing data by utilizing available information, a crucial benefit considering potential missing values within our dataset.

3. Implementing a basic Decision tree is notably uncomplicated, and the availability of diverse libraries and tools across programming languages further simplifies the utilization of Decision trees.

4. A notable strength of Decision trees is their lack of assumptions regarding data distribution. They adeptly manage both numerical and categorical data, circumventing the necessity for data to conform to normal distributions or exhibit linear relationships.

Features:
1. Our baseline model incorporates 'ANOMALY.LEVEL' and 'MONTH'. Both are quantitative variables, alleviating the need for encoding through feature transformers due to their inherent numerical nature. 
2. Using GridSearchCV, We can find the best max_depth, max_depth, criterion and min_samples_split of our model.
The output of the code is {'clf__criterion': 'entropy', 'clf__max_depth': 5, 'clf__min_samples_split': 2}, so we change the model combination and rerun the model.

**Step 2.3**: Test the F1-score on the testing data

In order to test the accuracy of our model, we decide to use F1-score to test testing data. If the f1-score is close to 1, which means that our model make a good prediction on the testing data. 

Our model train on 'ANOMALY.LEVEL' and 'MONTH' features and using the train_test_split to split the training and testing data to test the F1-score. We also use GridSearchCV to find the best combination of our model. So base on the two quantitative features we add to the model, the F1-Score of our baseline model is about 0.805, which is not bad. I will determine our model is good enough which only including two features.

### Final Model
Base on the beseline model, we plan to add another two other features 'OUTAGE.DURATION' and 'CUSTOMERS.AFFECTED'. Both of them are quantitative columns and sometimes with some really big outliers, so we decide to use log function transformer to decrease the variance of the data.
1. In our prediction both 'OUTAGE.DURATION' and 'CUSTOMERS.AFFECTED' are strongly relative to severe weather. Our prediction posits that if a power outage is triggered by severe weather conditions, the outage duration is likely to be prolonged. This extension in time may result from challenges faced by workers in repairing infrastructure during severe weather, necessitating waiting periods for improved weather conditions, thereby elongating the outage duration. If the outage duration become longer, which means that there will be more people affected, so we predict that the ourtage that causeby severe weather might have more people to be affected. 
2. Since we believe that both of the new features have the strong relationship, we believe that both of the features will help to improve our model's performance from the perspective of the data generating process

**Step 3.1**: Define the final model with the new columns and apply function transformers.

We keep the same model as the basic line model but add a log function transformer for the both of the new features and use the same DecisionTreeClassfier with the same hyperparameters we search last time. 

**Step 3.2**: Test the F1-score on the testing data befoe we search for a best hyperparameter combination. 

We calculate the F1-Score, which is much better than the basicline model. Our next step is to find the best combination of the hyperparameters to have a best prediction on our data.

**Step 3.3**: Perform a search using GridSearchCV to find the best hyperparameters of DicisionTreeClassifier.

We try to tune the max_depth, min_samples_split, and criterion as our hyperparameters. These three is the most comment hyperparameter of the decistion tree. 
- We know that if the max depth of a tree is large, it will most likely to ourfit the training data and we did not want that. 
- Also the min_samples_split, if we split it too much, it will also make the model too complicate which will also cause overfitting. 
- And criterion use different way to split and calculate in the tree, so we need to find the best way to make our data. 

**Step 3.4**: Test the F1-score on the testing data after we apply the best hyperparameter combination. 

Our F1-Score increases about 0.02 after we using the hyperparameter we find. so we will use this combination of hyperparameter to do the rest of the analysis.

**Step 3.5**:Confusion matrix

In order to actually see whether our model can make a good prediction, we decide to make a confusion matrix to plot the matrix directly.

The confusion matrix shows that there are 52 true positives (TP), 6 false negatives (FN), 133 true negatives (TN), and 10 false positives (FP). These values provide a detailed breakdown of the model’s predictions, allowing for a deeper analysis of its performance.

**Step 3.6**: Conclusion

Since the F1-score of the final model improve about 10% based on the baseline model, we conclude that our Final Model’s have a better performance over our Baseline Models' performance. It can effectively classify wether a power outage occurs by severe whehter. 

### Fairness Analysis

#### Cold episodes by season V.S. Warm episodes by season

The question we are trying to answer is whether the outages happened when cold episodes by season contribute to a higher frequency than Outages happened when warm episodes by season.

- **Group X**: "Cold episodes by season": 'ANOMALY.LEVEL' less than 0.

- **Group Y**: "Warm episodes by season": 'ANOMALY.LEVEL' larger and equal than 0.

- **Null Hypothesis**: Our model is fair. Its f1-score for Outages happened when cold episodes by season and Outages happened when warm episodes by season are roughly the same, and any differences are due to random chance.

- **Alternative Hypothesis**: Our model is unfair. Its f1-score for Outages happened when cold episodes by season is higher than its f1-score for Outages happened when warm episodes by season.

- **Test Statistic**: The difference in groups f1-score. (f1X - f1Y)

- **Significance level**: 0.01

### Conclusion

- The p-value is around 0.29, it is greater than the significance level of 0.01. 
- We fail to reject the null hypothesis. 
- There is not enough evidence to conclude that the precision for Outages happened when cold episodes by season is higher than its precision for Outages happened when warm episodes by season.

### Summary

For our fairness assessment, we have categorized the test dataset into two groups: 'ANOMALY.LEVEL' less than 0, and ‘ANOMALY.LEVEL’ larger and equal than 0.  'ANOMALY.LEVEL' less than 0 indicate the power outage happened when cold episodes by season. Our primary evaluation metric is F1-score. We propose a null hypothesis asserting that our model’s accuracy for determining outage happened with cold/warm episodes by season is roughly equivalent across all cases, with any observed differences attributable to random variability. Conversely, our alternative hypothesis suggests that the model demonstrates unfairness, with a higher f1-score for Outages happened when cold episodes by season. We have selected the f1-score disparity between those two groups as our test statistic, with a significance level of 0.01. After running a permutation test 1,000 times, we obtained a p-value of 0.29, which exceeds our significance level. This outcome leads us to retain the null hypothesis, indicating that our model, based on this f1-score, is fair. However, we cannot definitively assert its complete fairness as the permutation test results are also contingent on random chance. Hence, we recommend further testing with more data to verify if it is ‘truly fair’.