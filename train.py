import torch

# Train function
def train_model(model, dataloader, optimizer, criterion, acc_fn, device):
    
    # 0. Set the model to train mode
    model.train()
    
    # Initialize loss and accuracy
    train_loss, train_acc = 0, 0
    
    # Iterate throught the data
    for X, y in dataloader:
        
        # Move the data to the current device
        X, y = X.to(device), y.to(device)
        
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
    avg_acc /= len(dataloader)
    return avg_loss

# Test Function
def test_model(model, dataloader, optimizer, criterion, acc_fn, device):
    






    
    