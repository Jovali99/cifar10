from dataset_handler import get_dataloaders
import torch
import torch.nn.functional as F
import torchvision
import optuna

# Define the datasets
train_dataset = None
test_dataset = None

DEVICE = None

def train_one_epoch(model, optimizer, train_loader, device):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return correct / total

def objective(trial):
    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.8, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    T_max = trial.suggest_int("T_max", 20, 50)

    train_loader, val_loader = get_dataloaders(batch_size, train_dataset, test_dataset)

    model = torchvision.models.resnet18(num_classes=10).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    max_epochs = 50
    for epoch in range(max_epochs):
        train_one_epoch(model, optimizer, train_loader, DEVICE)
        scheduler.step()
        val_accuracy = evaluate(model, val_loader, DEVICE)
        trial.report(val_accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_accuracy
