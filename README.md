
## Introduction/About
This project aims to predict the popularity of songs using Spotify audio features. It involves:
- Data exploration
- Preprocessing
- Application of machine learning models like Linear Regression and Random Forest Regression
- The dataset contains various audio features provided by Spotify for a large number of songs
##Problem definition:
- Can we build a model to predict how popular (0-100 based on streams) a song will be on Spotify using
audio features like danceability, energy, and speechiness, among others?
## Data Exploration and Preprocessing
This phase includes:
- Importing necessary libraries such as Pandas, Seaborn, and Matplotlib for data manipulation and
visualization
- Loading the dataset and examining basic information like column names, data types, and missing values
- Creating various visualizations, including histograms and scatter plots, to understand the distribution of
popularity scores and the relationships between audio features and popularity
## Models Used
1. Linear Regression:
- An attempt is made to predict song popularity using Linear Regression
- The dataset is split into training and testing sets, and the model is trained and evaluated
- Despite efforts, the initial model yields low R^2 values, indicating poor predictive performance
2. Linear Regression with undersampling:
- Due to the unbalanced nature of the dataset, undersampling is performed to address the imbalance
between popular and unpopular songs
- The process involves randomly sampling a subset of less popular songs to achieve a balanced dataset
3. Random Forest Regression:
- A Random Forest Regression model is implemented to predict song popularity after undersampling
- Hyperparameter tuning is performed using Grid Search Cross-Validation to optimize model performance
- The best model is trained and evaluated, showing improved predictive performance compared to Linear
Regression
### Conclusion
The key points from the conclusion are:
- The dataset exhibits a skewed distribution of popularity scores, with the majority of songs having low
scores, making it challenging for linear regression to effectively predict popularity
- Many features show limited correlation with the target variable, further complicating the predictive task for
linear regression
- Undersampling the dataset was essential to address the imbalance, aiming to expose the model to more
relevant data points
- However, even with undersampling, linear regression failed to achieve satisfactory predictive performance- Implementing a Random Forest regression model proved to be a more effective approach, especially after
undersampling
- This model demonstrated superior predictive performance compared to linear regression, indicating its
suitability for this prediction task
- Grid Search Cross-Validation was instrumental in optimizing the Random Forest model's performance,
showcasing the importance of hyperparameter tuning in machine learning model optimization
- Comparing the performance of linear regression and Random Forest models highlighted the latter's
superiority, particularly in handling complex relationships and nonlinearities in the data
### What Have We Learned?
The key learnings from the project include:
- Proficiency in exploring and visualizing data using Pandas, Seaborn, and Matplotlib
- Skills in training regression models like Linear Regression using Statsmodels and Scikit-learn
- Addressing data imbalance with resampling techniques from Scikit-learn
- Hyperparameter tuning and model optimization with GridSearchCV
- Creating interactive plots with Plotly for effective data presentation
- Using Git and GitHub for version control and collaborative project management
- Exploring statistical analysis with NumPy and SciPy
- Leveraging external data sources through API integration
- Experimenting with neural networks using Keras and TensorFlow
- Enhanced understanding of machine learning concepts, programming skills, and real-world data analysis
experience
### References 
● Kaggle resampling strategies:

https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets

● Learning from imbalanced classes:

https://www.kdnuggets.com/2016/08/learning-from-imbalanced-classes.html/2

● Machine Learning Mastery - Neural Network Tutorial:

https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

Linear Regression:

● Scikit-learn documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
● Wikipedia article:
https://en.wikipedia.org/wiki/Ordinary_least_squares
● Towards Data Science article:
https://towardsdatascience.com/an-introduction-to-linear-regression-for-data-science-9056bbcdf6
75
Random Forest Regression:
● Scikit-learn documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
● Machine Learning Mastery: 
https://machinelearningmastery.com/
● Kaggle Learn course:
https://www.kaggle.com/code/prashant111/random-forest-classifier-tutorial
GridSearchCV:
● Scikit-learn documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
● Machine Learning Crash Course:
https://developers.google.com/machine-learning/crash-course
● Analytics Vidhya article:
https://towardsdatascience.com/gridsearchcv-for-beginners-db48a90114ee
