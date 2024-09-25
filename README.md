# CS 506 Final Project

### Members: Jonah, Cam, Sean

---

## Proposal

***Description of the project:***

- A project to predict cloud-based attacks using access logs and anomaly detection. The goal would be to identify potential security breaches in cloud environments (such as AWS, Azure, or GCP) by analyzing access logs and spotting unusual patterns, such as unauthorized access attempts or data exfiltration. 

***Goals:***

- Develop a model that will predict, given the nature of the request, whether or not it can be deemed “evil” (confirmed malicious activity).

***What data needs to be collected and how we will collect it:***

- Data about web requests sent to cloud providers that indicates whether such a request was confirmed to be a part of an attack. Such a dataset can be retrieved straight from the cloud providers, as such data is logged. We will pull public dataset(s) using the requests library in python. 

***How we plan on modeling the data:***

- We plan to use machine learning models (including but not limited to logistic regression, decision trees, random forests, SVM, naive Bayes, and neural networks) trained on x% of the dataset to determine if a given request is malicious. The exact model we implement will depend on our data analysis. 

***How we plan on visualizing the data:***

- Initially, we will perform exploratory data analysis (EDA) using scatter plots, bar charts, heatmaps, importance of features plot in decision trees and other visualizations to identify which features are correlated.  This will help guide our feature selection for the model.

***Test plan:***

- In the context of training and testing, we will use x% of the oldest data in the dataset to train the model, and the remaining “new” data to test and check if, given the nature of the request, such a request is “evil.” We are going to use the time column to split the data for a progressive analysis as if the data is coming in real-time. 

---