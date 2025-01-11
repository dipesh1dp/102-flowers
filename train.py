import torch
from tqdm.auto import tqdm

# Train function
def train_model(model, dataloader, optimizer, criterion, acc_fn, device):
    # 0. Set the model to train mode
    model.train()
    
    # Initialize loss and accuracy
    train_loss, train_acc = 0, 0
    
    # Iterate throught the data
    for X, y in dataloader:
        # Move the data to the current device
        X, y = X.to(device), y.long().to(device)
        
        # 1. Forward Propagation
        y_hat = model(X)
        
        # 2. Calculate the loss
        loss = criterion(y_hat, y)
        train_loss += loss.item()

        # Calculate accuracy
        y_preds = torch.argmax(torch.softmax(y_hat, dim=1), dim=1)
        acc = acc_fn(y_preds, y)
        train_acc += acc

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Back Propagation
        loss.backward()

        # 5. Step the optimizer
        optimizer.step()

    # Calcuate and return average loss and accuracy
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


# Test Function
def val_model(model, dataloader, optimizer, criterion, acc_fn, device):
    # Initialize loss and accuracy
    val_loss, val_acc = 0, 0
    
    # 0. Set the model to evaluation mode to turn off setting not needed for validation
    model.eval()

    with torch.inference_mode():
        # Iterate throught the data
        for X, y in dataloader:
            
            # Move the data to the current device
            X, y = X.to(device), y.long().to(device)
            
            # 1. Forward Propagation
            y_hat = model(X)
    
            # 2. Calculate the loss
            loss = criterion(y_hat, y)
            val_loss += loss
            

            # Calculate accuracy
            val_pred = torch.argmax(torch.softmax(y_hat, dim=1), dim=1)
            acc = acc_fn(val_pred, y)
            val_acc += acc

        # Calcuate and return average loss and accuracy
        val_loss /= len(dataloader)
        val_acc /= len(dataloader)

        return val_loss, val_acc
    
    
# Evaluation Function
def eval_model(model, dataloader, optimizer, criterion, acc_fn, device):
    # Initialize loss and accuracy
    total_loss, total_acc = 0, 0
    
    # 0. Set the model to evaluation mode to turn off setting not needed for validation
    model.eval()

    with torch.inference_mode():
        # Iterate throught the data
        for X, y in dataloader:

            # Move the data to the current device
            X, y = X.to(device), y.long().to(device)
            
            # 1. Forward Propagation
            y_hat = model(X)
    
            # 2. Calculate the loss
            loss = criterion(y_hat, y)
            total_loss += loss.item()
            

            # Calculate accuracy
            preds = torch.argmax(torch.softmax(y_hat, dim=1), dim=1)
            acc = acc_fn(preds, y)
            total_acc += acc.item()

        # Calcuate and return average loss and accuracy
        total_loss /= len(dataloader)
        total_acc /= len(dataloader)

        return {'Loss': {total_loss}, 'Accuracy': {total_acc}}





    
    