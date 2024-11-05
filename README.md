# CS 506 Final Project

### Members: Jonah Gluck (jonahg@bu.edu), Yanjia Kan(kyjbu@bu.edu), Sean McCarty(mccartys@bu.edu)

---

## Proposal

***Description of the project:***

- A project to predict cloud-based attacks using access logs and anomaly detection. The goal would be to identify potential security breaches in cloud environments (such as AWS, Azure, or GCP) by analyzing access logs and spotting unusual patterns, such as unauthorized access attempts or data exfiltration. 

***Goals:***

- Develop a model that will predict, given the nature of the request, whether or not it can be deemed “evil” (confirmed malicious activity).This will be achieved through analyzing the relationships between features using data analysis techniques, ultimately building a predictive model.


***What data needs to be collected and how we will collect it:***

- Data about web requests sent to cloud providers that indicates whether such a request was confirmed to be a part of an attack. Such a dataset can be retrieved straight from the cloud providers, as such data is logged. We will pull public dataset(s) using the requests library in python. 

***How we plan on modeling the data:***

- We plan to use machine learning models (including but not limited to logistic regression, decision trees, random forests, SVM, naive Bayes, and neural networks) trained on x% of the dataset to determine if a given request is malicious. The exact model we implement will depend on our data analysis. 

***How we plan on visualizing the data:***

- Initially, we will perform exploratory data analysis (EDA) using scatter plots, bar charts, heatmaps, importance of features plot in decision trees and other visualizations to identify which features are correlated.  This will help guide our feature selection for the model.

***Test plan:***

- In the context of training and testing, we will use x% of the oldest data in the dataset to train the model, and the remaining “new” data to test and validate and check if, given the nature of the request, such a request is “evil.” We are going to use the time column to split the data for a progressive analysis as if the data is coming in real-time. Depending on the size of the dataset, we may also consider using k-fold cross-validation to improve model accuracy.  Metrics such as cross-entropy loss, confusion matrices, and others will be used to evaluate the model's performance on the test set.

---

## Midterm Report

Click [here](https://youtu.be/eP1ZI6YIXi0) for presentation

***Data used:***

- The dataset we used is the public [LUFlow Dataset]([https://www.kaggle.com/path-to-dataset](https://www.kaggle.com/datasets/mryanm/luflow-network-intrusion-detection-data-set)) from Kaggle, which is a flow-based intrusion detection dataset. LUFlow contains telemetry containing emerging attack vectors through the composition of honeypots within Lancaster University's address space. The labelling mechanism is autonomous and is supported by a robust ground truth through correlation with third part Cyber Threat Intelligence (CTI) sources, enabling the constant capture, labelling and publishing of telemetry to this repository. The dataset contains 15 features and three types of labels: benign, outlier, and malicious. 
- To thoroughly understand the dataset, our project has so far focused on the data from June 2022 for analysis and model training.

***Visualizations & insights:***

- TODO: Include plots & describe relationships + how they might relate to the classifications 
To better understand our dataset, we plan to include a series of visualizations, such as scatter plots, heatmaps, and bar charts. These plots will help uncover relationships between features and identify patterns indicative of different classifications (benign, outlier, malicious). Insights from these visualizations will guide our feature engineering process, as well as help us pinpoint key attributes that may signal malicious behavior.

***First attempt at modeling data:***

- TODO: Talk about what deep learning methods were used
Our initial modeling approach utilized a deep learning neural network with four layers (128, 64, 32 neurons) and ReLU activation functions between layers. We used a Cross-Entropy loss function optimized with Adam (learning rate: 0.001) and trained over 100 epochs. This model setup provided a baseline to evaluate the dataset's suitability for deep learning.

***Preliminary results:***

- TODO: Talk about accuracy of model etc
Our first model achieved a test accuracy of 75.85%, with class-specific AUC scores of 0.91 for benign, 0.82 for outlier, and 0.91 for malicious classes. The confusion matrix indicated areas where the model struggled, particularly in distinguishing between outliers and malicious events. These preliminary results provide a foundation for further improvements, including feature engineering and hyperparameter tuning.

***Next steps:***

- Going further, we will apply feature engineering for data processing, including data cleaning, data balancing, and feature correlation analysis.
- We will also experiment with various machine learning models such as KNN, decision trees, and random forests.
- Additionally, we plan to explore deep learning models like CNN and ResNet.
