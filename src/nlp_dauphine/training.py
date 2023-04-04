import torch


def train_epoch(model, opt, criterion, dataloader, device):
    """
    Implement a training function, which will train the model with the corresponding optimizer and criterion,
    with the appropriate dataloader, for one epoch.

    Arguments
    ---------
        model: torch.Model
            Neural Network to train
        opt: torch.optim
            Optimizer used to train the network (Adam, RMSProp...)
        criterion: torch.nn
            Loss used to assess the model
        dataloader: torch.utils.data.DataLoader
            Dataset
        device: str
            Device used to train the network (cpu/gpu)

    Returns
    -------
        losses: list
            List of the losses per epoch
    """
    model.train()
    losses = []
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        # (1) Forward
        pred = model.forward(x)
        # (2) Compute the loss
        loss = criterion(pred, y)
        # (3) Compute gradients with the criterion
        loss.backward()
        # (4) Update weights with the optimizer
        opt.step()
        losses.append(loss.item())
        # Count the number of correct predictions in the batch - here, you'll need to use the sigmoid
        num_corrects = (torch.round(torch.sigmoid(pred)) == y).float().sum()
        acc = 100.0 * num_corrects / len(y)

        if i % 20 == 0:
            print(
                "Batch "
                + str(i)
                + " : training loss = "
                + str(loss.item())
                + "; training acc = "
                + str(acc.item())
            )
    return losses


def eval_model(model, criterion, evalloader, device):
    """
    Eval function of the model (without the criterion)

    Arguments
    ---------
        model: torch.Model
            Neural Network to train
        opt: torch.optim
            Optimizer used to train the network (Adam, RMSProp...)
        evalloader: torch.utils.data.DataLoader
            Dataset
        device: str
            Device used to train the network (cpu/gpu)

    Returns
    -------
        losses: list
            List of the losses per epoch
    """
    model.eval()
    total_epoch_loss = 0
    total_epoch_acc = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(evalloader):
            x, y = x.to(device), y.to(device)
            pred = model.forward(x)
            loss = criterion(pred, y)
            num_corrects = (torch.round(torch.sigmoid(pred)) == y).float().sum()
            acc = 100.0 * num_corrects / len(y)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss / (i + 1), total_epoch_acc / (i + 1)


def experiment(
    model,
    opt,
    criterion,
    training_dataloader,
    valid_dataloader,
    test_dataloader,
    num_epochs=5,
    early_stopping=True,
):
    """
    A function which will help you execute experiments rapidly - with a early_stopping option when necessary.

    Arguments
    ---------
        model: torch.Model
            Neural Network to train
        opt: torch.optim
            Optimizer used to train the network (Adam, RMSProp...)
        criterion: torch.nn
            Loss used to assess the model
        training_dataloader: torch.utils.data.DataLoader
            Training Dataset
        valid_dataloader: torch.utils.data.DataLoader
            Valid Dataset
        test_dataloader: torch.utils.data.DataLoader
            Test Dataset
        device: str
            Device used to train the network (cpu/gpu)
        num_epochs: int
            Number of epochs
        early_stopping: bool
            Stop the training if no learning before the end of the epoch

    Returns
    -------
        losses: list
            List of the losses per epoch
    """
    train_losses = []
    if early_stopping:
        best_valid_loss = 10.0
    print("Beginning training...")
    for e in range(num_epochs):
        print("Epoch " + str(e + 1) + ":")
        train_losses += train_epoch(model, opt, criterion, training_dataloader)
        valid_loss, valid_acc = eval_model(model, criterion, valid_dataloader)
        print(
            "Epoch "
            + str(e + 1)
            + " : Validation loss = "
            + str(valid_loss)
            + "; Validation acc = "
            + str(valid_acc)
        )
        if early_stopping:
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
            else:
                print("Early stopping.")
                break
    test_loss, test_acc = eval_model(model, criterion, test_dataloader)
    print(
        "Epoch "
        + str(e + 1)
        + " : Test loss = "
        + str(test_loss)
        + "; Test acc = "
        + str(test_acc)
    )
    return train_losses
