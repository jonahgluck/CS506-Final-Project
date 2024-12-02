import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_and_process_data():
    df1 = pd.read_csv('data/2022.06.12.csv')
    df2 = pd.read_csv('data/2022.06.13.csv')
    df3 = pd.read_csv('data/2022.06.14.csv')

    df_dataset = pd.concat([df1, df2, df3])
    df_dataset.reset_index(drop=True, inplace=True)

    df_dataset = df_dataset.drop(['time_start', 'time_end', 'dest_ip', 'src_ip'], axis=1)

    df_dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

    df_dataset.dropna(inplace=True)
    df_dataset.drop_duplicates(inplace=True)

    label_mapping = {'benign': 0, 'outlier': 'malicious', 'malicious': 1}
    df_dataset['label'] = df_dataset['label'].replace(label_mapping).astype(int)

    return df_dataset

def scale_and_split(df_dataset):
    numerical_columns = ['avg_ipt', 'bytes_in', 'bytes_out', 'dest_port', 'entropy',
                         'num_pkts_out', 'num_pkts_in', 'src_port', 'total_entropy', 'duration']

    scaler = MinMaxScaler()
    df_dataset[numerical_columns] = scaler.fit_transform(df_dataset[numerical_columns])

    train, test = train_test_split(df_dataset, test_size=0.2, random_state=42)

    X_train = train[numerical_columns].values
    y_train = train['label'].values
    X_test = test[numerical_columns].values
    y_test = test['label'].values

    return X_train, X_test, y_train, y_test, train, test

def kmeans_clustering(X_train, y_train):
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)

    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(X_train)

    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=cluster_labels, cmap='viridis')
    plt.title('K-means Clustering Results')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter)

    plt.subplot(122)
    scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis')
    plt.title('Actual Labels')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter)

    plt.tight_layout()
    plt.show()

    accuracy = np.mean(cluster_labels == y_train)
    print(f"Clustering accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    df_dataset = load_and_process_data()
    X_train, X_test, y_train, y_test, train, test = scale_and_split(df_dataset)

    print("Dataset processed. Beginning clustering analysis...")
    kmeans_clustering(X_train, y_train)

