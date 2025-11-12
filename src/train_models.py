from src.cifar_handler import CifarInputHandler
from LeakPro.leakpro import LeakPro
from models.resnet18_model import ResNet18
import torch
import torch.nn.functional as F
from torch import save, load, optim, nn
import os
import pickle

def trainTargetModel(cfg, train_loader, test_loader, train_indices, test_indices):
    print("-- Training model ResNet18 on cifar10  --")
    os.makedirs("target", exist_ok=True)

    if(cfg["data"]["dataset"] == "cifar10"):
        num_classes = 10
    else:
        raise ValueError(f"Incorrect dataset {cfg['data']['dataset']}, should be cifar10")

    model = ResNet18(num_classes=num_classes)

    """Parse training configuration"""
    lr = cfg["train"]["learning_rate"]
    weight_decay = cfg["train"]["weight_decay"]
    epochs = cfg["train"]["epochs"]
    momentum = cfg["train"]["momentum"]
    t_max = cfg["train"]["t_max"]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,)

    # --- Initialize scheduler ---
    if t_max is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    else:
        scheduler = None

    train_result = CifarInputHandler().train(dataloader=train_loader,
                                             model=model,
                                             criterion=criterion,
                                             optimizer=optimizer,
                                             epochs=epochs,
                                             scheduler=scheduler)

    test_result = CifarInputHandler().eval(test_loader, model, criterion)

    model.to("cpu")
    save(model.state_dict(), os.path.join(cfg["run"]["log_dir"], "target_model.pkl"))
    
    # Create and Save LeakPro metadata
    meta_data = LeakPro.make_mia_metadata(train_result = train_result,
                                      optimizer = optimizer,
                                      loss_fn = criterion,
                                      dataloader = train_loader,
                                      test_result = test_result,
                                      epochs = epochs,
                                      train_indices = train_indices,
                                      test_indices = test_indices,
                                      dataset_name = cfg["data"]["dataset"])
    with open("target/model_metadata.pkl", "wb") as f:
        pickle.dump(meta_data, f)

    return train_result, test_result
