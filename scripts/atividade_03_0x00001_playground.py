import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from IPython.display import display

plt.style.use('fivethirtyeight')

class MLP(nn.Module):
    def __init__(
            self,
            architecture,
            activation='relu', 
            dropout_p=None,
            use_l2=False
        ):
        """
        `todo`
        """
        super(MLP, self).__init__()
        #
        self.n_layers = len(architecture) - 1
        self.layers = nn.ModuleList()
        self.dropout = None if dropout_p is None else nn.Dropout(dropout_p)
        self.use_l2 = use_l2
        #
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("função de ativação deve ser 'relu' ou 'tanh'")
        #
        for i in range(self.n_layers):
            self.layers.append(
                nn.Linear(architecture[i], architecture[i+1])
            )
            #
            nn.init.normal_(self.layers[i].weight, mean=0.0, std=1)
            nn.init.zeros_(self.layers[i].bias)
    
    def forward(self, x):
        """
        `todo`
        """
        for i in range(self.n_layers - 1):
            x = self.layers[i](x)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)
        #
        x = self.layers[-1](x)
        return x
    
def get_loss_function(task='multiclass'):
    if task.lower() == 'binary':
        return nn.BCEWithLogitsLoss()
    elif task.lower() == 'multiclass':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError("task must be 'binary' or 'multiclass'")


def get_optimizer(model, optimizer_type='sgd', lr=0.01, momentum=0.9, weight_decay=0):
    if optimizer_type.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("optimizer_type must be 'sgd' or 'adam'")


def get_output_size(y, task='binary'):
    """
    `todo`
    """
    if task.lower() == 'binary':
        return 1  # One output neuron for binary classification
    else:
        # For multiclass, return the number of unique classes
        return len(np.unique(y))
    
def train_model(
        model,
        dataloader_train,
        dataloader_val,
        criterion,
        optimizer,
        epochs=100,
        patience=20,
        device='cpu',
        l2_lambda=0.0,
        task='multiclass'
    ):
    """
    `todo`
    """
    model.to(device)
    history = {
        'train_loss': list(),
        'train_acc': list(),
        'val_loss': list(),
        'val_acc': list()
    }
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None
    #
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, targets in dataloader_train:
            inputs, targets = inputs.to(device), targets.to(device)
            # forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Handle different problem types
            if task.lower() == 'binary':
                targets = targets.float()  # Ensure targets are float for BCE
                loss = criterion(outputs, targets)
                predicted = (outputs > 0.0).float()
            else:  # multiclass
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs, 1)
                
            # regularization
            if model.use_l2:
                l2_reg = 0.0
                for param in model.parameters():
                    l2_reg += torch.norm(param, 2)
                loss += l2_lambda * l2_reg
            # backpropagation
            loss.backward()
            optimizer.step()
            # stats (train)
            train_loss += loss.item() * inputs.size(0)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        # Calcular métricas de treino
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        # eval
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in dataloader_val:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # Handle different problem types
                if task.lower() == 'binary':
                    targets = targets.float()  # Ensure targets are float for BCE
                    loss = criterion(outputs, targets)
                    predicted = (outputs > 0.0).float()
                else:  # multiclass
                    loss = criterion(outputs, targets)
                    _, predicted = torch.max(outputs, 1)
                    
                # stats
                val_loss += loss.item() * inputs.size(0)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        # stats (eval)
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        #
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        #
        print(f'Época {epoch+1}/{epochs} - Treino: perda={train_loss:.4f}, acc={train_acc:.4f} | Validação: perda={val_loss:.4f}, acc={val_acc:.4f}')
        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping na época {epoch+1}')
                break
    #
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history

def evaluate_model(model, dataloader_test, criterion, device='cpu', task='multiclass'):
    """
    `todo`
    """
    model.to(device)
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in dataloader_test:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Handle different problem types
            if task.lower() == 'binary':
                targets = targets.float().view(-1, 1)
                loss = criterion(outputs, targets)
                predicted = (outputs > 0.0).float()
            else:  # multiclass
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs, 1)
                
            # stats
            test_loss += loss.item() * inputs.size(0)
            test_total += targets.size(0)
            test_correct += (predicted == targets).sum().item()
    #
    test_loss = test_loss / test_total
    test_acc = test_correct / test_total
    
    return test_loss, test_acc

def prepare_classification_data(X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, val_size=0.2, batch_size=32, task='binary', convert_labels=False):
    """
    Prepares classification data for PyTorch training.
    
    Args:
        X_train: Training feature data
        y_train: Training labels
        X_val: Validation feature data (optional)
        y_val: Validation labels (optional)
        X_test: Test feature data (optional)
        y_test: Test labels (optional)
        val_size: Validation size if X_val and y_val not provided
        batch_size: Batch size for DataLoader
        task: 'binary' or 'multiclass'
        convert_labels: Whether to convert -1/1 labels to 0/1 for BCE loss
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Handle validation split if validation data not provided
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=val_size,
            random_state=42, 
            stratify=y_train
        )
    
    # Handle test split if test data not provided
    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, 
            test_size=0.2,
            random_state=42, 
            stratify=y_train
        )
    
    # Convert -1/1 labels to 0/1 for BCE loss if needed
    if convert_labels and task.lower() == 'binary':
        y_train = (y_train == 1).astype(int)  # Convert -1 to 0, keep 1 as 1
        y_val = (y_val == 1).astype(int)
        y_test = (y_test == 1).astype(int)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Handle different task types
    if task.lower() == 'binary':
        # For binary classification with BCE loss
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
        y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)
        y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    else:
        # For multiclass classification with CrossEntropy loss
        y_train_tensor = torch.LongTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val)
        y_test_tensor = torch.LongTensor(y_test)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


train = pd.read_csv("../data/atividade_03/train_dataset.csv")
validation = pd.read_csv("../data/atividade_03/validation_dataset.csv")
test = pd.read_csv("../data/atividade_03/test_dataset.csv")

X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values
X_val = validation.iloc[:, :-1].values
y_val = validation.iloc[:, -1].values
X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

task = 'multiclass'
input_features = X_train.shape[1]
output_size = get_output_size(y_train, task=task)

train_loader, val_loader, test_loader = prepare_classification_data(
    X_train, 
    y_train, 
    X_val=X_val,
    y_val=y_val,
    X_test=X_test, 
    y_test=y_test,
    batch_size=32,
    task=task,
    convert_labels=True  # Convert -1/1 labels to 0/1 for BCE loss if binary
)

architecture = [input_features, 64, 16, output_size]

model = MLP(architecture, activation='tanh', dropout_p=0.2, use_l2=True)

criterion = get_loss_function(task=task)

optimizer = get_optimizer(model, optimizer_type='adam', lr=0.001, momentum=0.09)

history = train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,  # Using SGD optimizer here
    epochs=1000,
    patience=100,
    l2_lambda=0.001,
    task=task
)

# Evaluate the model
test_loss, test_acc = evaluate_model(
    model, 
    test_loader, 
    criterion, 
    task=task
)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")