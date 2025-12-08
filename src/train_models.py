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

def trainShadowModel(args):
    smh, i, indx, data_indices, shadow_population = args

    # Dataloader for this worker
    data_loader = smh.handler.get_dataloader(data_indices, params=None)

    # Model blueprint
    model, criterion, optimizer = smh._get_model_criterion_optimizer()
    print(f"Training shadow model: {indx}")
    # Train model
    training_results = smh.handler.train(data_loader, model, criterion, optimizer, smh.epochs)
    shadow_model = training_results.model

    # Evaluate
    remaining_indices = list(set(shadow_population) - set(data_indices))
    dataset_params = data_loader.dataset.return_params()
    test_loader = smh.handler.get_dataloader(remaining_indices, params=dataset_params)
    test_result = smh.handler.eval(test_loader, shadow_model, criterion)

    return indx, shadow_model, training_results, test_result

def worker_wrapper(args, result_queue):
    """
    args: (smh, i, indx, data_indices, shadow_population, gpu_id)
    result_queue: multiprocessing.Queue for returning small results
    """
    smh, i, indx, data_indices, shadow_population, gpu_id = args

    # Bind this process to a single visible GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Now set torch device index 0 (the first visible device)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # Run training
    try:
        idx, shadow_model, training_results, test_result = trainShadowModel((smh, i, indx, data_indices, shadow_population))
        # Save the state dict & metadata inside worker (avoid pickling model)
        state_path = f"{smh.storage_path}/{smh.model_storage_name}_{idx}.pth"
        meta_path = f"{smh.storage_path}/{smh.metadata_storage_name}_{idx}.pkl"

        torch.save(shadow_model.state_dict(), state_path)

        # Prepare metadata (keep it small)
        meta = {
            "index": idx,
            "train_metrics": training_results.metrics,
            "test_result": test_result,
            "state_path": state_path,
            "meta_path": meta_path
        }
        # Optionally save meta to disk
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        # push small result into queue
        result_queue.put((idx, state_path, meta_path))

    except Exception as e:
        result_queue.put(("error", str(e)))
        raise
    
def create_shadow_models_parallel(smh, num_models, shadow_population, training_fraction, gpus):
    # compute indices, assignments as in original function
    data_size = int(len(shadow_population)*training_fraction)
    all_indices, filtered_indices = smh._filter(data_size)
    n_existing_models = len(filtered_indices)
    indices_to_use = []
    next_index = max(all_indices) + 1 if all_indices else 0
    while len(indices_to_use) < (num_models - n_existing_models):
        indices_to_use.append(next_index); next_index += 1

    A = smh.construct_balanced_assignments(len(shadow_population), num_models)
    shadow_population = np.array(shadow_population)

    # build tasks (same as original loop)
    tasks = []
    for i, indx in enumerate(indices_to_use):
        data_indices = shadow_population[np.where(A[i, :] == 1)]
        gpu_id = gpus[i % len(gpus)]  # round-robin
        tasks.append((smh, i, indx, data_indices, shadow_population, gpu_id))

    # Launch processes (limit concurrency to number of GPUs)
    procs = []
    result_q = mp.Queue()
    for args in tasks:
        p = mp.Process(target=worker_wrapper, args=(args, result_q))
        p.start()
        procs.append(p)

    # Wait for processes and collect results
    results = []
    for p in procs:
        p.join()

    while True:
        try:
            results.append(result_q.get_nowait())
        except Empty:
            break

    return results