import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tvm
from torchvision.datasets import ImageNet


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')


def topk_ct(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k)
        return res


def validate_model(model, valid_dl, loss_f):
    "Compute performance of the model on the validation dataset"
    val_loss = 0.0
    top1_ct = 0
    top5_ct = 0
    image_ct = 0
    model.eval()
    with torch.inference_mode():
        for i, data in enumerate(valid_dl):
            inputs = data[0].to(DEVICE)
            targets = data[1].to(DEVICE)

            outputs = model(inputs)    
            batch_size = len(inputs)
            image_ct += batch_size

            batch_val_loss = loss_f(outputs, targets).item()
            val_loss += batch_val_loss * batch_size

            # Compute accuracy and accumulate
            top1_bct, top5_bct = topk_ct(outputs, targets, topk=(1, 5))
            top1_ct += top1_bct.item()
            top5_ct += top5_bct.item()
    
    return val_loss / image_ct, top1_ct / image_ct, top5_ct / image_ct


def get_model_imagnet_acc(model, imn_dir, max_batch=20, eval_per=5):
    """
    Train the last layer of the model on ImageNet and return the best top-1 and top-5 accuracy
    achieved on the validation set.
    args:
        model: a torchvision model, assume the last layer is called model.fc
        imn_dir: the directory containing the ImageNet dataset
        max_batch: the maximum number of batches to train
        eval_per: the number of batches between each validation evaluation
    """
    model = model.to(DEVICE)

    # only train the last layer
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.fc.parameters():
        param.requires_grad_(True)

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()

    IMN_transform = tvm.ResNet18_Weights.IMAGENET1K_V1.transforms()
    train_dset = ImageNet(root=imn_dir, split='train', transform=IMN_transform)
    test_dset = ImageNet(root=imn_dir, split='val', transform=IMN_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dset,
                                               batch_size=128,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=test_dset,
                                              batch_size=128,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=8)

    # initialize
    batch_n = 0  # the numbder of batches the model has trained on so far
    sample_ct = 0  # number of training samples the model has trained on so far
    best_object_acc1 = 0.0
    best_object_acc5 = 0.0

    # Train the model
    model.train()
    while batch_n < max_batch:
        for data in train_loader:
            inputs = data[0].to(DEVICE)
            targets = data[1].to(DEVICE)

            outputs = model(inputs)
            train_loss = loss_func(outputs, targets)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            batch_n += 1
            sample_ct += len(inputs)
            
            if batch_n % eval_per == 0:
                # validate model
                val_loss, val_acc1, val_acc5 = validate_model(model, test_loader, loss_func)
                model.train()

                out_string = f"Batch Number: {batch_n:10d}, Train Loss: {train_loss.item():.3f}, Valid Loss: {val_loss:.3f}"
                if val_acc1 > best_object_acc1:
                    best_object_acc1 = val_acc1
                if val_acc5 > best_object_acc5:
                    best_object_acc5 = val_acc5
                out_string += f", Val Top1Acc: {val_acc1:.3f}"
                out_string += f", Val Top5Acc: {val_acc5:.3f}"
                print(out_string)
            
            if batch_n >= max_batch:
                break
    
    return best_object_acc1, best_object_acc5
