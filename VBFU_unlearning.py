#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import torch.nn.functional as F
import copy
from experiment import Experiment
from server import Server
from client import Clients

def non_intrusive_verification(clients, forget_clients, model_before, model_after, device, kappa=0.3, fine_tune_iters=5, delta=0.1):
    model_before.eval()
    model_after.eval()
    model_before.to(device)
    model_after.to(device)
    forget_data, forget_labels = get_client_data_from_loader(clients, forget_clients, device)
    if len(forget_data) == 0:
        print("No forget data available for verification.")
        return False

    print("Marking Phase:")
    with torch.no_grad():
        outputs = model_before(forget_data)
        losses = F.cross_entropy(outputs, forget_labels, reduction='none')

        num_samples = len(forget_data)
        k = int(kappa * num_samples)
        _, high_loss_indices = torch.topk(losses, k)

        preds = outputs.argmax(dim=1)
        misclassified_indices = (preds != forget_labels).nonzero(as_tuple=True)[0]
        misclassified_indices = misclassified_indices[:k]

        marker_indices = torch.unique(torch.cat([high_loss_indices, misclassified_indices]))
        D_m_data = forget_data[marker_indices]
        D_m_labels = forget_labels[marker_indices]

        print(f"Selected {len(marker_indices)} marker samples (high-loss and misclassified).")

    model_tuned = copy.deepcopy(model_before)
    model_tuned.train()
    optimizer = torch.optim.SGD(model_tuned.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(fine_tune_iters):
        optimizer.zero_grad()
        outputs = model_tuned(D_m_data)
        loss = criterion(outputs, D_m_labels)
        loss.backward()
        optimizer.step()

    print("Checking Phase:")

    def compute_loss_variance(model, data, labels):
        with torch.no_grad():
            outputs = model(data)
            losses = F.cross_entropy(outputs, labels, reduction='none')
            mean_loss = losses.mean()
            variance = ((losses - mean_loss) ** 2).mean().item()
        return variance

    phi_before = compute_loss_variance(model_before, D_m_data, D_m_labels)
    phi_after = compute_loss_variance(model_after, D_m_data, D_m_labels)
    delta_phi = phi_after - phi_before

    print(f"Loss Variance Before: {phi_before:.4f}")
    print(f"Loss Variance After: {phi_after:.4f}")
    print(f"Delta Phi: {delta_phi:.4f} (Threshold: {delta})")

    verification_passed = delta_phi > delta
    print(f"Verification {'Passed' if verification_passed else 'Failed'}")
    return verification_passed

def vbfu_unlearning(exp: Experiment, server: Server, clients: Clients):
    print('Policy and aggregations: ', server.policy, exp.n_aggregs)

    psi_increments = []
    t_total, t_start = 0, 0

    if exp.limit_train_iter:
        iter_max = int(exp.limit_train_iter * server.train_length)

    model_before_unlearning = copy.deepcopy(server.g_model)

    for r, (W_r, n_aggreg) in enumerate(zip(server.policy, exp.n_aggregs)):
        print('Round r: ', r)

        server.forget_Wr(W_r)

        T_loss_acc = 20

        for t in range(t_start, n_aggreg):
            server.g_model.to(server.device)

            if exp.limit_train_iter:
                print(exp.limit_train_iter, t, t_start, iter_max, W_r)

            if exp.limit_train_iter and t > iter_max and W_r != []:
                print("Stopping learning because of time constraint")
                break

            if t % T_loss_acc == 0:
                loss, acc = server.loss_acc_global(clients)
                if np.isnan(loss):
                    break
                if acc >= server.stop_acc and t >= 50:
                    if server.compute_diff and server.unlearn_scheme == "VBFU":
                        if t % 100 == 0:
                            break
                    else:
                        break
                elif acc >= server.stop_acc - 0.5:
                    T_loss_acc = 1
                elif acc >= server.stop_acc - 1.:
                    T_loss_acc = min(T_loss_acc, 2)
                elif acc >= server.stop_acc - 2.5:
                    T_loss_acc = min(T_loss_acc, 5)
                elif acc >= server.stop_acc - 5.:
                    T_loss_acc = min(T_loss_acc, 10)

            working_clients = server.sample_clients()
            num_clients = len(working_clients)

            local_models = clients.local_work(
                working_clients,
                server.g_model,
                server.loss_f,
                server.lr_l,
                server.n_SGD,
                server.lambd,
                server.clip,
                server.compute_diff
            )

            if server.compute_diff:
                local_models, local_grads = local_models

            server.aggregation(local_models, server.g_model)

            server.compute_metric(working_clients, local_models)

            server.keep_best_model()

        if W_r == []:
            iter_max = int(exp.limit_train_iter * server.t)
        if server.train_length == 0 and server.unlearn_scheme != "train":
            server.train_length = server.t

        server.loss_acc_global(clients)

        if server.unlearn_scheme == 'train':
            print('Saving best models')
            exp.save_best_models(f"{exp.file_name}_{server.r}", server.best_models[-1])

        print("\n")

    verification_result = non_intrusive_verification(
        clients, W_r, model_before_unlearning, server.g_model, server.device,
        kappa=0.1, fine_tune_iters=5, delta=0.1
    )
    print(f"Non-Intrusive Verification Result: {'Success' if verification_result else 'Failure'}")