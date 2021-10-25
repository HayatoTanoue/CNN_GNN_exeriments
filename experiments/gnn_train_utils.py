# cv_train
def cv_train(model, loader, device, criterion, optimizer):
    correct = 0
    running_loss = 0
    model.train()
    for data in loader:  # Iterate in batches over the training dataset.
        data.to(device)
        out = model(data)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        running_loss += loss.item()

        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.

    train_loss = running_loss / len(loader)
    train_acc = correct / len(loader.dataset)

    return train_loss, train_acc


# cv_valid
def cv_test(model, loader, device, criterion):
    model.eval()
    correct = 0
    running_loss = 0

    for data in loader:
        data.to(device)
        out = model(data)  # Perform a single forward pass.
        loss = criterion(out, data.y)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.

        running_loss += loss.item()

    val_loss = running_loss / len(loader)
    val_acc = correct / len(loader.dataset)  # Derive ratio of correct predictions.

    return val_loss, val_acc


def gnn_train_val_1epoch(
    model, train_loader, valid_loader, device, optimizer, criterion
):
    """train and valid 1poch"""
    train_loss, train_acc = cv_train(model, train_loader, device, criterion, optimizer)
    val_loss, val_acc = cv_test(model, valid_loader, device, criterion)
    return train_loss, train_acc, val_loss, val_acc
