Problem statement - Implement multiple classification models -
Build an interactive Streamlit web application to demonstrate your models - Deploy
the app on Streamlit Community Cloud 


Dataset description - This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended.

The table below presents the evaluation metrics of all implemented classification models on the Mushroom dataset.  


| ML Model Name           | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
|-------------------------|----------|-----------|-----------|--------|----------|-----------|
| Logistic Regression     | 0.9491   | 0.9816    | 0.9493    | 0.9491 | 0.9491   | 0.8982    |
| Decision Tree           | 0.9984   | 0.9983    | 0.9984    | 0.9984 | 0.9984   | 0.9967    |
| KNN                     | 0.9951   | 1.0000    | 0.9951    | 0.9951 | 0.9951   | 0.9902    |
| Naive Bayes.            | 0.9282   | 0.9543    | 0.9282    | 0.9282 | 0.9282   | 0.8562    |
| Random Forest           | 1.0000   | 1.0000    | 1.0000    | 1.0000 | 1.0000   | 1.0000    |
| XGBoost                 | 0.9984   | 1.0000    | 0.9984    | 0.9984 | 0.9984   | 0.9967    |



Model Performance Observations

The following table summarizes the performance behavior of each machine learning model on the Mushroom dataset.

| ML Model Name            | Observation about Model Performance  |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression      | Showed good but comparatively lower performance. As a linear model, it may not fully capture complex non-linear relationships present in the data. |
| Decision Tree            | Performed extremely well with near-perfect accuracy. Slightly lower than Random Forest, indicating minor overfitting or variance compared to the ensemble approach. |
| KNN                      | Achieved very high accuracy and perfect AUC. Performance indicates that similar feature patterns strongly cluster together in the dataset. Slightly sensitive to local variations. |
| Naive Bayes              | Recorded the lowest performance among all models. The independence assumption between features likely limited its effectiveness for this dataset. |
| Random Forest            | Achieved perfect classification performance across all evaluation metrics (Accuracy, AUC, F1, MCC). The ensemble of multiple decision trees effectively captured complex feature interactions in the dataset. |
| XGBoost                  | Delivered performance comparable to Decision Tree with perfect AUC score. The boosting mechanism enhanced predictive strength and reduced bias effectively. |



