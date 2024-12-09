# CS 506 Final Project

### Members: Jonah Gluck (jonahg@bu.edu), Yanjia Kan(kyjbu@bu.edu), Sean McCarty(mccartys@bu.edu)
```
project/
├── data/
│   ├── 2022.06.12.csv
│   ├── 2022.06.13.csv
│   ├── 2022.06.14.csv
├── *.py files
├── Makefile
├── requirements.txt
```

Prerequisites

Install Python 3.7 or higher.
Clone the repository and ensure the data/ folder contains the required .csv files.

Installation

Install the necessary dependencies using pip:
```
make install
```

or 
```
pip install -r requirements.txt
```

Run Instructions
Best model for randomforest:
```
make run_rf
```

Simple Linear Model for Classification
To run the simple linear model for binary classification:
```
make run_simple
```

This script:

Uses a fully connected neural network to classify the preprocessed data.
Outputs training loss and accuracy on the test set.
ResNet-Inspired Model for Classification
To run the ResNet-inspired model for binary classification:
```
make run_resnet
```
This script:

Implements a ResNet-like architecture adapted for tabular data.
Outputs training loss and accuracy on the test set.

K-Means Model for Classification
To run the K-Means model for binary classification:
```
make run_kmeans
```

Cleaning Temporary Files

To clean up temporary files and cache:  
```
make clean
```

Dataset Information

The dataset contains network traffic features such as packet size, protocol, and entropy, along with labels for binary classification:

    Benign (0)
    Malicious (1)

Dependencies

Dependencies are listed in the requirements.txt file. Key libraries include:

    numpy and pandas for data manipulation.
    matplotlib for data visualization.
    scikit-learn for preprocessing, clustering, and evaluation.
    torch for implementing neural networks.

Example Output

    KMeans Clustering:
        Visualizes clusters based on PCA-reduced features.
        Reports clustering accuracy.

    Model Training:
        Logs training progress.
        Outputs final accuracy and confusion matrices.

Notes

Ensure that the data/ folder exists and contains the required .csv files. Modify the file paths in the scripts if the dataset structure changes.

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

- To better understand our dataset, we plan to include a series of visualizations, such as scatter plots, heatmaps, and bar charts. These plots will help uncover relationships between features and identify patterns indicative of different classifications (benign, outlier, malicious). Insights from these visualizations will guide our feature engineering process, as well as help us pinpoint key attributes that may signal malicious behavior.
- One of the major insights was the fact that per feature, the difference between benign & malicious were apparent, while outliers seemed to be more unpredictable, meaning that a careful treatment of outliers will aid in the development of our model.  
#### Classification counts over three days:
<img src="/readme_images/labels_6:12.png" alt="drawing" width="375"/>
<img src="/readme_images/labels_6:13.png" alt="drawing" width="375"/>
<img src="/readme_images/labels_6:14.png" alt="drawing" width="375"/>

#### Average bytes in over three days per classification:
<img src="/readme_images/bytes_in_6:12.png" alt="drawing" width="375"/>
<img src="/readme_images/bytes_in_6:13.png" alt="drawing" width="375"/>
<img src="/readme_images/bytes_in_6:14.png" alt="drawing" width="375"/>

#### Average entropy over three days per classification
<img src="/readme_images/entropy_6:12.png" alt="drawing" width="375"/>
<img src="/readme_images/entropy_6:13.png" alt="drawing" width="375"/>
<img src="/readme_images/entropy_6:14.png" alt="drawing" width="375"/>

- ***Bar charts for other labels shown in presentation & sean-data-exploration notebook***

***First attempt at modeling data:***

- Our initial modeling approach utilized a deep learning neural network with four layers (128, 64, 32 neurons) and ReLU activation functions between layers. We used a Cross-Entropy loss function optimized with Adam (learning rate: 0.001) and trained over 100 epochs. This model setup provided a baseline to evaluate the dataset's suitability for deep learning.

***Preliminary results:***

- Our first model achieved a test accuracy of 75.85%, with class-specific AUC scores of 0.91 for benign, 0.82 for outlier, and 0.91 for malicious classes. The confusion matrix indicated areas where the model struggled, particularly in distinguishing between outliers and malicious events. These preliminary results provide a foundation for further improvements, including feature engineering and hyperparameter tuning.

***Next steps:***

- Going further, we will apply feature engineering for data processing, including data cleaning, data balancing, and feature correlation analysis.
- We will also experiment with various machine learning models such as KNN, decision trees, and random forests.
- Additionally, we plan to explore deep learning models like CNN and ResNet.

# Final Report
## Dataset

## Feature Engineering
I'll help you create a concise yet informative feature engineering section for a README based on the document. I'll focus on presenting the key feature engineering techniques and their purposes in a clear, readable format.

# Feature Engineering

Our feature engineering approach transforms raw network flow data into a rich, informative representation that enables more sophisticated machine learning models. We employed several strategic techniques to enhance model predictive capabilities:

## 1. Ratio-Based Features

Ratio features reveal complex relationships between network traffic attributes:

- **Byte Ratio**: Compares incoming vs. outgoing bytes to identify data transfer patterns
- **Packet Ratio**: Analyzes the balance of incoming and outgoing packets
- **Byte-Packet Ratios**: Calculates average bytes per packet for both incoming and outgoing traffic
- **Flow Efficiency**: Measures overall data transmission efficiency by comparing total bytes to total packets

## 2. Entropy-Based Features

Entropy features help detect anomalous or unusual network behaviors:

- **Entropy per Byte**: Normalizes data randomness relative to total bytes transferred
- **Total Entropy Ratio**: Provides another perspective on data randomness
- **Bidirectional Entropy Ratio**: Compares entropy across different flow directions

## 3. Binary Indicator Features

Simple binary flags highlight critical network conditions:

- **Well-Known Port Indicators**: Flag flows originating from or targeting standard service ports (ports < 1024)

## 4. Difference-Based Features

Difference calculations expose imbalances in network traffic:

- **Byte Difference**: Net byte flow indicating data receive/send dynamics
- **Packet Difference**: Net packet flow revealing packet transmission balance

## 5. Logarithmic Transformations

Log transformations stabilize variance and improve model training:

- **Log Transformations**: Applied to bytes, duration to reduce the impact of extreme values and make patterns more discernible

## 6. Aggregation Features

Comprehensive features providing holistic network flow insights:

- **Total Bytes**: Aggregate incoming and outgoing data volume
- **Total Packets**: Combined packet count across flow directions
- **Average Bytes per Packet**: Efficiency metric for data transmission

## Results

These feature engineering techniques enabled:
- Capturing complex network traffic relationships
- Normalizing and stabilizing data representations
- Identifying critical network conditions
- Improving model accuracy (up to 94% for Improved Linear Model and CNN)

## Next Steps

Continued exploration of domain-specific feature transformations and integration of additional data sources promises further improvements in predictive capabilities.

## Models
In our journey to accurately predict outcomes using the LuFlow network dataset, we experimented with various modeling approaches, blending traditional machine learning techniques with modern deep learning methods. While each approach offered unique insights and improvements, the Random Forest classifier ultimately delivered the highest accuracy. Here's a breakdown of our efforts and findings:

***Machine Learning***
**KMeans**
Initially, we applied the KMeans algorithm to perform clustering analysis on the dataset. To simplify observation, outliers and malignant cases were grouped into label 1, while benign cases were assigned to label 0. Principal Component Analysis (PCA) was employed for dimensionality reduction, and a clustering plot based on the first and second principal components was generated. The clustering accuracy was 0.54. The plot demonstrated a significant overlap between the two clusters, making it challenging to distinguish between them. Subsequently, we attempted clustering using the first and second most important features, yielding an accuracy of only 0.48. The high degree of overlap persisted, making it difficult to establish an effective decision boundary.

**Baseline Linear Model**
Our first attempt was a simple neural network with one hidden layer—a straightforward approach to establish a performance benchmark.
Performance: Initially struggled with ~75% accuracy.


**Deep Learning**
Encouraged by the potential of deep learning, we ventured into building neural network models to see if they could surpass the performance of Random Forests.

Takeaway: The baseline model was too simplistic to capture the intricate patterns within the data, leading to lower accuracy. It also missed key functionality such as batch normalization and dropout. Our "ImprovedLinearModel" boosted accuracy to 94% on our validation set. This model added more layers, incorporated batch normalization, and implemented dropout for regularization. The 1D CNN matched the performance of the Improved Linear Model, indicating that while convolutional layers can be powerful, they didn't provide additional benefits over the enhanced fully connected architecture for our specific dataset. 、

**Final Model -- Random Forest**
For model selection, we primarily considered accuracy and F1 score as evaluation metrics. The final model selected was the Random Forest classifier. This choice was based on the complexity of the dataset and its suitability for multi-class classification tasks. Random Forest was preferred because it can automatically capture feature interactions, and each tree considers different feature combinations, enhancing model robustness.
In terms of optimization, grid search and cross-validation were applied. We focused on tuning four key hyperparameters:
- The number of trees (n_estimators),
- The maximum depth of the trees (max_depth),
- The minimum number of samples required to split a node (min_samples_split), and
- The minimum number of samples required to be at a leaf node (min_samples_leaf).
The first two parameters increase model complexity and enhance performance, while the latter two control tree growth to prevent overfitting. The final hyperparameter configuration was: {'max_depth': 20, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 200}.

## Results

## Conclusion
