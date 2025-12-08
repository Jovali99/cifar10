from src.cifar_handler import CifarInputHandler
from LeakPro.leakpro import LeakPro
from src.models.resnet18_model import ResNet18
from src.models.wideresnet28_model import WideResNet
import numpy as np
import torch
import torch.nn.functional as F
from torch import save, load, optim, nn
import os
import pickle
import multiprocessing as mp
from queue import Empty

from LeakPro.leakpro.schemas import LeakProConfig
from LeakPro.leakpro.attacks.mia_attacks.lira import AttackLiRA
from LeakPro.leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from LeakPro.leakpro.input_handler.mia_handler import MIAHandler
from src.cifar_handler import CifarInputHandler

def trainTargetModel(cfg, train_loader, test_loader, train_indices, test_indices):
    print("-- Training model ResNet18 on cifar10  --")
    os.makedirs("target", exist_ok=True)

    if(cfg["data"]["dataset"] == "cifar10"):
        num_classes = 10
    elif(cfg["data"]["dataset"] == "cifar100"):
        num_classes = 100
    else:
        raise ValueError(f"Incorrect dataset {cfg['data']['dataset']}, should be cifar10, cifar 100 or cinic10")

    if cfg["train"]["model"] == "resnet":
        model = ResNet18(num_classes=num_classes)
        print("Training resnet")
    elif cfg["train"]["model"] == "wideresnet":
        drop_rate = cfg["train"]["drop_rate"]
        model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10, dropRate=drop_rate)
        print("Training wideresnet")

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
    metadata_pkl_path = os.path.join(cfg["run"]["log_dir"], "model_metadata.pkl")
    with open(metadata_pkl_path, "wb") as f:
        pickle.dump(meta_data, f)

    return train_result, test_result

def trainFbDTargetModel(cfg, train_loader, test_loader, train_indices, test_indices, fbd_cfg, mia_type: str):
    print("-- Training model ResNet18 on cifar10  --")
    os.makedirs("target", exist_ok=True)

    if(cfg["data"]["dataset"] == "cifar10"):
        num_classes = 10
    elif(cfg["data"]["dataset"] == "cifar100"):
        num_classes = 100
    else:
        raise ValueError(f"Incorrect dataset {cfg['data']['dataset']}, should be cifar10, cifar 100 or cinic10")

    model = ResNet18(num_classes=num_classes)

    """Parse training configuration"""
    lr = cfg["train"]["learning_rate"]
    weight_decay = cfg["train"]["weight_decay"]
    epochs = cfg["train"]["epochs"]
    momentum = cfg["train"]["momentum"]
    t_max = cfg["train"]["t_max"]
    noise_std = fbd_cfg["noise_std"]

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,)

    # --- Initialize scheduler ---
    if t_max is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    else:
        scheduler = None

    train_result = CifarInputHandler().trainFbD(dataloader=train_loader,
                                             model=model,
                                             criterion=criterion,
                                             optimizer=optimizer,
                                             epochs=epochs,
                                             noise_std=noise_std,
                                             scheduler=scheduler)

    test_result = CifarInputHandler().eval(test_loader, model, criterion)

    model.to("cpu")
    model_name = mia_type+"_fbd_target_model.pkl"
    save(model.state_dict(), os.path.join(cfg["run"]["log_dir"], model_name))
    
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
    metadata_name = mia_type+"_fbd_model_metadata.pkl"
    metadata_pkl_path = os.path.join(cfg["run"]["log_dir"], metadata_name)
    with open(metadata_pkl_path, "wb") as f:
        pickle.dump(meta_data, f)

    return model, train_result, test_result
    
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass
    
def create_shadow_models_parallel(audit_config, train_config, gpu_ids):
    leakpro_configs = LeakProConfig(**audit_config)
    handler = MIAHandler(leakpro_configs, CifarInputHandler)
    configs = handler.configs.audit.attack_list[0]

    num_shadow_models = configs["num_shadow_models"]
    online = configs["online"]

    attack = AttackLiRA(handler=handler, configs=configs)
    attack_data_indices = attack.sample_indices_from_population(
        include_train_indices=online,
        include_test_indices=online
    )

    m = len(attack_data_indices)

    # 1. Construct full assignment ONCE
    full_A = ShadowModelHandler(handler).construct_balanced_assignments(
        m,
        num_shadow_models
    )


    # 2. Split the model rows among GPUs
    model_ids = np.arange(num_shadow_models)
    model_splits = np.array_split(model_ids, len(gpu_ids))

    procs = []
    for gpu_id, model_subset in zip(gpu_ids, model_splits):

        A_slice = full_A[model_subset]  # Only rows needed for that GPU

        p = mp.Process(
            target=sm_worker,
            args=(
                audit_config,
                train_config,
                gpu_id,
                list(model_subset),
                A_slice,
                attack_data_indices
            )
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
    return

def sm_worker(audit_config, train_config, gpu_id, model_indices, A_slice, attack_data_indices): 
    torch.cuda.set_device(gpu_id)
    
    print(f"Sm training started on gpu: {gpu_id}")
    
    leakpro_configs = LeakProConfig(**audit_config)
    handler = MIAHandler(leakpro_configs, CifarInputHandler)
    configs = handler.configs.audit.attack_list[0]

    attack = AttackLiRA(handler=handler, configs=configs)
    online = configs["online"]
    training_data_fraction = attack.training_data_fraction

    smh = ShadowModelHandler(handler)
    smh.epochs = train_config["train"]["epochs"]
    smh.batch_size = train_config["train"]["batch_size"]
    smh.learning_rate = train_config["train"]["learning_rate"]
    smh.momentum = train_config["train"]["momentum"]

    # Only train models belonging to this worker
    smh.create_shadow_models(
        num_models=len(model_indices),
        shadow_population=attack_data_indices,
        training_fraction=training_data_fraction,
        online=online,
        model_indices=model_indices,
        assignment=A_slice

    )