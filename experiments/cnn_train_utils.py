import torch


def cv_train(model, train_loader, device, optimizer, criterion):
    """交差検証訓練"""
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predict = torch.max(outputs.data, 1)
        correct += (predict == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    return train_loss, train_acc


def cv_valid(model, valid_loader, device, criterion):
    """交差検証検証"""
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(valid_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predict = torch.max(outputs.data, 1)
            correct += (predict == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(valid_loader)
    val_acc = correct / total

    return val_loss, val_acc


def cnn_train_val_1epoch(
    model, train_loader, valid_loader, device, optimizer, criterion
):
    """train and valid 1poch"""
    train_loss, train_acc = cv_train(model, train_loader, device, optimizer, criterion)
    val_loss, val_acc = cv_valid(model, valid_loader, device, criterion)
    return train_loss, train_acc, val_loss, val_acc
