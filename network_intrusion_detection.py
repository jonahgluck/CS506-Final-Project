import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Load and preprocess data
def load_data():
    df1 = pd.read_csv('data/2022.06.12.csv')
    df2 = pd.read_csv('data/2022.06.13.csv')
    df3 = pd.read_csv('data/2022.06.14.csv')

    df_dataset = pd.concat([df1, df2, df3])
    df_dataset.reset_index(drop=True, inplace=True)
    df_dataset = df_dataset.drop(['num_pkts_out', 'num_pkts_in'], axis=1)
    df_dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_dataset.dropna(inplace=True)
    df_dataset.drop_duplicates(inplace=True)

    # Map labels to integers
    label_mapping = {'benign': 2, 'outlier': 0, 'malicious': 1}
    df_dataset['label'] = df_dataset['label'].map(label_mapping)

    return df_dataset

def preprocess_data(df_dataset):
    X = df_dataset.drop(['label', 'dest_ip', 'src_ip', 'dest_port', 'src_port', 'time_start', 'time_end'], axis=1).values
    y = df_dataset['label'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y = torch.tensor(y, dtype=torch.long)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
class SimpleLinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleLinearModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, output_size)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x

def train_and_evaluate(X_train, X_test, y_train, y_test, input_size, output_size, epochs=100, lr=0.001):
    model = SimpleLinearModel(input_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)

    loss_values = []

    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    plt.figure()
    plt.plot(range(epochs), loss_values, label="Training Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()

    with torch.no_grad():
        outputs = model(X_test)
        probabilities = softmax(outputs, dim=1).cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        predictions = np.argmax(probabilities, axis=1)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test_np, predictions)
    cm_display = ConfusionMatrixDisplay(conf_matrix, display_labels=['benign', 'outlier', 'malicious'])
    cm_display.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    plt.figure()
    for i in range(output_size):
        binary_true = (y_test_np == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Class')
    plt.legend(loc="lower right")
    plt.show()

    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
        print(f'Accuracy on test set: {accuracy * 100:.2f}%')

# Main script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    input_size = X_train.shape[1]
    output_size = 3

    train_and_evaluate(X_train, X_test, y_train, y_test, input_size, output_size)

