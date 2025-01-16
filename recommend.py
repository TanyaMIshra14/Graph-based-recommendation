!pip install torch-geometric
import pandas as pd 
import os 
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, HeteroData
from itertools import combinations
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.decomposition import PCA
import networkx as nx
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import to_networkx

#Preprocess and loading function
#We create a function to load and preprocess the csv data so that it could be further processed to convert it into graph data 
#we sort the data based on the customer ID and drop the rows where the customer IDs are not available 
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    data = data.sort_values(by="CustomerID", ascending=False)
    data = data.dropna(subset=['CustomerID'])
    return data

# Padding Function
#We create a padding function so that if in case there is any issue with the dimensionality of the data and output after we convert it into graph it can be avoided 
def pad_features(tensor, target_size):
    current_size = tensor.size(1)
    if current_size < target_size:
        padding = torch.zeros((tensor.size(0), target_size - current_size), dtype=tensor.dtype)
        return torch.cat([tensor, padding], dim=1)
    return tensor
  # Customer Features 
#Before converting the csv data into graph data we need to extract features,labels,eges and then create the nodes.
#The nodes will be of two types so will be the features i.e. customer nodes and their features and product nodes and their features.
#This function extracts the features of the customer nodes like quantity of the product they bought ,total money spent and country or location.We also index them tom avoid the discrepencies in deciding which one to refer first.
def create_customer_features(data):
    customer_features = data.groupby('CustomerID').agg(
        total_purchases=('Quantity', 'sum'),
        total_spent=('Quantity', lambda x: (x * data.loc[x.index, 'UnitPrice']).sum()),
        country=('Country', 'first')
    ).reset_index()
    country_dummies = pd.get_dummies(customer_features['country'], prefix='country')
    customer_features = pd.concat([customer_features, country_dummies], axis=1)
    customer_features = customer_features.drop(columns=['country'])
    customer_to_index = {customer: i for i, customer in enumerate(customer_features['CustomerID'])}
    customer_node_features = torch.tensor(customer_features.drop(columns='CustomerID').values, dtype=torch.float)
    customer_node_features = pad_features(customer_node_features, target_size=39)
    return customer_node_features, customer_to_index

# Product Features 
#This function extracts the features of the prodccut nodes like average quantity of the product  bought ,average cost of the unit product, and description.We also index them tom avoid the discrepencies in deciding which one to refer first.
def create_product_features(data):
    product_features = data.groupby('StockCode').agg(
        average_quantity_sold=('Quantity', 'mean'),
        average_unit_price=('UnitPrice', 'mean'),
        description=('Description', 'first')
    ).reset_index()
    product_features['description_length'] = product_features['description'].apply(len)
    product_features = product_features.drop(columns=['description'])
    product_to_index = {product: i for i, product in enumerate(product_features['StockCode'])}
    product_node_features = torch.tensor(product_features.drop(columns='StockCode').values, dtype=torch.float)
    product_node_features = pad_features(product_node_features, target_size=39) 
    return product_node_features, product_to_index
# Customer Labels
#Now we need to label the customer nodes to identify the type of customers so that accordingly the products can be recommended to them.We group the users based on the customer ID and find the aggreate of the total money they spent and how many unique items they purchases and also what was the frequency of the purchase,based on these we classy the users into ocassional and loyal.
def create_customer_labels(data):
    customer_metrics = data.groupby('CustomerID').agg(
        total_spent=('Quantity', lambda x: (x * data.loc[x.index, 'UnitPrice']).sum()),
        purchase_frequency=('InvoiceNo', 'nunique')
    ).reset_index()
    def label_customers(row):
        if row['total_spent'] > 500:
            return 'loyal'
        elif 100 < row['total_spent'] <= 500:
            return 'occasional'
        else:
            return 'new'
    customer_metrics['customer_label'] = customer_metrics.apply(label_customers, axis=1)
    customer_labels = torch.tensor(customer_metrics['customer_label'].astype('category').cat.codes.values, dtype=torch.long)
    return customer_labels

# Creating the Edges
#Nodes in the data will be of 2 types customers and products and so will be the edges which basically show the relationship between the nodes and these edges will be assigned weights that will determine the importance of the relationship.
#This function creates the customer-product and product-product edges.
def create_edges(data):
    customer_product_edges = data.groupby(['CustomerID', 'StockCode']).agg(
        total_quantity=('Quantity', 'sum')
    ).reset_index()
    customer_ids = customer_product_edges['CustomerID'].astype('category').cat.codes.values
    stock_codes = customer_product_edges['StockCode'].astype('category').cat.codes.values
    weights = customer_product_edges['total_quantity'].values
    customer_product_edge_index = torch.tensor([customer_ids, stock_codes], dtype=torch.long)
    customer_product_edge_weights = torch.tensor(weights, dtype=torch.float)
    product_product_edges = []
    for invoice, group in data.groupby('InvoiceNo'):
        product_list = group['StockCode'].tolist()
        for product1, product2 in combinations(product_list, 2):
            product_product_edges.append((product1, product2))
    product_product_edges_df = pd.DataFrame(product_product_edges, columns=['Product1', 'Product2'])
    product_product_edges_count = product_product_edges_df.groupby(['Product1', 'Product2']).size().reset_index(name='co_purchase_count')
    product1_ids = product_product_edges_count['Product1'].astype('category').cat.codes.values
    product2_ids = product_product_edges_count['Product2'].astype('category').cat.codes.values
    co_purchase_weights = product_product_edges_count['co_purchase_count'].values
    product_product_edge_index = torch.tensor([product1_ids, product2_ids], dtype=torch.long)
    product_product_edge_weights = torch.tensor(co_purchase_weights, dtype=torch.float)
    return (customer_product_edge_index, customer_product_edge_weights, 
            product_product_edge_index, product_product_edge_weights)

# Preparing the Graph data
#Now with the help of extracted features,labels and edges we build the graph data that will be fed as the input for the graph based learning algorithms to be implemented .
def prepare_data(file_path):
    data = load_and_preprocess_data(file_path)
    customer_node_features, customer_to_index = create_customer_features(data)
    product_node_features, product_to_index = create_product_features(data)
    num_customer_nodes = customer_node_features.size(0)
    num_product_nodes = product_node_features.size(0)
    total_nodes = num_customer_nodes + num_product_nodes
    node_features = torch.cat([customer_node_features, product_node_features], dim=0)
    customer_product_edge_index, customer_product_edge_weights, product_product_edge_index, product_product_edge_weights = create_edges(data)
    adjusted_product_edge_index = product_product_edge_index + num_customer_nodes
    edge_index = torch.cat([customer_product_edge_index, adjusted_product_edge_index], dim=1)
    edge_weights = torch.cat([customer_product_edge_weights, product_product_edge_weights], dim=0)
    customer_labels = create_customer_labels(data)
    graph_data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weights)
    graph_data.y = torch.cat([customer_labels, torch.full((num_product_nodes,), -1, dtype=torch.long)])
    return graph_data

# Model Building 
#We have selected graph convolutional networks as our model due to its efficient performance and accurracy as it utilises the concepts of deep learning.
#We define the class GNNModel and define the layers of the models 
class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        return x

# Training the model
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing the model
@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_pred = pred[data.test_mask]
    test_labels = data.y[data.test_mask]
    accuracy = accuracy_score(test_labels.cpu(), test_pred.cpu())
    f1 = f1_score(test_labels.cpu(), test_pred.cpu(), average='weighted')
    report = classification_report(test_labels.cpu(), test_pred.cpu(), target_names=['new', 'occasional', 'loyal'])
    return accuracy, f1, report

# Generating the recommendations 
def generate_recommendations(model, customer_id, customer_to_index, product_to_index, top_n=5):
    model.eval()
    with torch.no_grad():
        customer_idx = customer_to_index[customer_id]
        out = model(data.x, data.edge_index)
        customer_embedding = out[customer_idx]
        product_scores = out[len(customer_to_index):] 
        scores = (product_scores * customer_embedding).sum(dim=1)
        top_products = scores.topk(top_n).indices
        recommended_products = [list(product_to_index.keys())[i] for i in top_products]
    return recommended_product

def mean_average_precision(pred, true):
    return average_precision_score(true.cpu(), pred.cpu(), average='macro')

# split the data into training and testing
def split_data(graph_data, num_customer_nodes):
    num_nodes = graph_data.y.size(0)
    indices = torch.randperm(num_customer_nodes)
    train_split = int(num_customer_nodes * 0.8)
    graph_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    graph_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    graph_data.train_mask[indices[:train_split]] = True
    graph_data.test_mask[indices[train_split:num_customer_nodes]] = True
    return graph_data

# Visualise the node embeddings 
def visualize_nodes(data, model):
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
    pca = PCA(n_components=2)
    node_embeddings_2d = pca.fit_transform(embeddings.cpu())
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1], c=data.y.cpu(), cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, ticks=[0, 1, 2], label="Customer Label")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Node Embeddings Visualization")
    plt.show()

# Main Function
def main(file_path):
    data = prepare_data(file_path)
    num_customer_nodes = data.y[data.y != -1].size(0) 
    data = split_data(data, num_customer_nodes)
    model = GNNModel(in_channels=data.num_features, hidden_channels=64, out_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)  
    recommendation=generate_recommendations(model, customer_id, customer_to_index, product_to_index, top_n=5)
    epochs = 160
    for epoch in range(epochs):
        loss = train(model, data, optimizer, criterion)
        if epoch//10==16:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    visualize_nodes(data, model)

file_path = '/home/ec2-user/SageMaker/OnlineRetail.csv'
main(file_path)
