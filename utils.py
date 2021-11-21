import torch
import numpy as np


def train_loop(dataloader, val_loader, model, loss_fn, loss_bin, optimizer, device, epoch):
    # checkpoint = torch.load('model2.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    size = len(dataloader.dataset)
    running_loss = 0
    valid_loss_min = np.Inf
    # monitor training loss
    train_loss = 0.0
    valid_loss = 0.0
    model.train()
    for batch, (spec, y, ibm_mask, irm_mask) in enumerate(dataloader):
        spec = spec.to(device)
        spec = spec.float()
        y = y.to(device)
        y = y.float()
        ibm_mask = ibm_mask.to(device)
        ibm_mask = ibm_mask.float()
        irm_mask = irm_mask.to(device)
        irm_mask = irm_mask.float()

        # Compute prediction and loss
        pred_total, pred_spec, pred_ibm, pred_irm = model(spec)

        output_loss = loss_fn(pred_total, y)
        loss_spec = loss_fn(pred_spec, y)
        loss_ibm = loss_fn(pred_ibm, ibm_mask)
        loss_irm = loss_fn(pred_irm, irm_mask)

        total_loss = loss_spec + loss_ibm + loss_irm + output_loss

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()*spec.size(0)
        if batch % 100 == 0:
            tot_loss, current = total_loss.item(), batch * len(spec)
            t_loss, current = output_loss.item(), batch * len(spec)
            b_loss, current = loss_ibm.item(), batch * len(spec)
            r_loss, current = loss_irm.item(), batch * len(spec)
            s_loss, current = loss_spec.item(), batch * len(spec)
            print(f"total_loss: {tot_loss:>7f} , output_loss: {t_loss:>7f} , spec_loss: {s_loss:>7f} , ibm_loss: {b_loss:>7f} , irm_loss: {r_loss:>7f}   [{current:>5d}/{size:>5d}]")
            print(f"total_loss: {total_loss:>7f} [{current:>5d}/{size:>5d}]")
    model.eval() # prep model for evaluation
    for (spec, y, ibm_mask, irm_mask) in val_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        spec = spec.to(device)
        spec = spec.float()
        y = y.to(device)
        y = y.float()
        ibm_mask = ibm_mask.to(device)
        ibm_mask = ibm_mask.float()
        irm_mask = irm_mask.to(device)
        irm_mask = irm_mask.float()
        with torch.no_grad():
            pred_total_val, pred_spec_val, pred_ibm_val, pred_irm_val = model(spec)

        # calculate the loss
        output_loss_val = loss_fn(pred_total_val, y)
        loss_spec_val = loss_fn(pred_spec_val, y)
        loss_ibm_val = loss_fn(pred_ibm_val, ibm_mask)
        loss_irm_val = loss_fn(pred_irm_val, irm_mask)

        total_loss_val = output_loss_val + loss_ibm_val + loss_irm_val + loss_spec_val
        # update running validation loss
        valid_loss += total_loss_val.item()*spec.size(0)

    # print training/validation statistics
    # calculate average loss over an epoch
    train_loss = train_loss/len(dataloader.dataset)
    valid_loss = valid_loss/len(val_loader.dataset)

    if epoch % 1 == 0:
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch+1,
            train_loss,
            valid_loss
            ))


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for (X, y, ibm, irm) in dataloader:
            X = X.to(device)
            X = X.float()
            y = y.to(device)
            y = y.float()

            pred, clean, ibm, irm = model(X)

            # pred = torch.transpose(pred)
            test_loss += loss_fn(pred, y).item()
            # pred = torch.transpose(pred, 0 ,1)
            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
