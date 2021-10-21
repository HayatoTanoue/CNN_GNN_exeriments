# cv_train
def cv_train(model, loader, device, criterion, optimizer, model_name="GCN"):
    model.train()

    for data in loader:  # Iterate in batches over the training dataset.

        data.to(device)
        if model_name == "GCN":
            out = model(
                data.x, data.edge_index, data.batch
            )  # Perform a single forward pass.
        else:
            out = model(data)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


# cv_valid
def cv_test(model, loader, device, model_name="GCN"):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data.to(device)
        if model_name == "GCN":
            out = model(
                data.x, data.edge_index, data.batch
            )  # Perform a single forward pass.
        else:
            out = model(data)  # Perform a single forward pass.
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.

    val_acc = correct / len(loader.dataset)  # Derive ratio of correct predictions.
    return val_acc
