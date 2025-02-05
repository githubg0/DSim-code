import os
import torch
from sklearn import metrics
from sklearn.metrics import accuracy_score
import logging as log
from utils import batch_data_to_device

def train(model, loaders, args):
    log.info("training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_len = len(loaders['train'].dataset)
    
    best_auc = -10
    for epoch in range(args.n_epochs):
        loss_all = 0
        for step, data in enumerate(loaders['train']):
            with torch.no_grad():
                x, y = batch_data_to_device(data, args.device)
            model.train()
            
            loss = model.get_loss(x, y)
            loss_all += loss.item()
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            step += 1
            
            
        show_loss = loss_all / train_len
        acc_v, auc_v = evaluate(model, loaders['valid'], args)
        acc_t, auc_t = evaluate(model, loaders['test'], args)
        log.info('Epoch: {:03d}, Loss: {:.7f}, valid_acc: {:.7f}, valid_auc: {:.7f}, test_acc: {:.7f}, test_auc: {:.7f}'.format(epoch, show_loss, acc_v, auc_v, acc_t, auc_t))
        
        torch.save([args, model.cpu()], os.path.join(args.run_dir, 'params_latest.pt'))
        model = model.to(args.device)
        if auc_t > best_auc:
            torch.save([args, model.cpu()], os.path.join(args.run_dir, 'params_best.pt'))
            model = model.to(args.device)
            best_auc = auc_t


def evaluate(model, loader, args):
    model.eval()
    eval_sigmoid = torch.nn.Sigmoid()
    y_list, hat_y_list = [], []
    with torch.no_grad():
        for data in loader:
            x, y = batch_data_to_device(data, args.device)
       
            hat_y_prob = model.get_result(x)
            y_list.append(y)
            hat_y_list.append(eval_sigmoid(hat_y_prob))

    y_tensor = torch.cat(y_list, dim = 0).int()
    hat_y_prob_tensor = torch.cat(hat_y_list, dim = 0)
    acc = accuracy_score(y_tensor.cpu().numpy(), (hat_y_prob_tensor > 0.5).int().cpu().numpy())
    fpr, tpr, _ = metrics.roc_curve(y_tensor.cpu().numpy(), hat_y_prob_tensor.cpu().numpy(), pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return acc, auc


