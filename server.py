import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
from copy import deepcopy
from create_model import load_model
from branch import Branch
from client import Clients
import random
from privacy import psi, get_std
import pickle
from policy import policies
import copy


def noise_model(model: Module, sigma: float, device: torch.device = "cpu"):
    model.to(device)
    if sigma > 0:
        for w_k in model.parameters():
            noise = torch.normal(0.0, sigma, size=w_k.shape).to(device)
            w_k.data.add_(noise)


class Server:
    def __init__(self, args: dict, clients: Clients = None):
        self.dataset_name = args["dataset_name"]
        self.seed = args["seed"]
        self.unlearn_scheme = args["unlearn_scheme"]
        self.T = args["T"]
        self.M = args["M"]
        self.n_sampled = args["n_sampled"]
        self.lr_g = args["lr_g"]
        self.lr_l = args["lr_l"]
        self.n_SGD = args["n_SGD"]
        self.lambd = args["lambd"]
        self.epsilon = args["epsilon"]
        self.sigma = args["sigma"]
        self.psi_star = psi(self.sigma, self.epsilon, self.M)
        self.stop_acc = args["stop_acc"]
        self.compute_diff = args["compute_diff"]
        self.device = args["device"]
        self.clip = args["clip"]
        self.policy = [[]]
        self.train_length = 0
        self.model = args["model"]

        self.clients = clients
        self.forget_clients = set()

        self.policy = policies[args["forgetting"]]
        if self.unlearn_scheme == "train" and args["compute_diff"]:
            self.policy = policies[args["forgetting"]]

        if self.DP:
            self.model_clip = float(args["unlearn_scheme"].split("_")[1])
            self.sigma_DP = get_std(2 * self.model_clip / self.M, self.epsilon, self.M)

        self.model_0, self.loss_f, self.model_ul = load_model(self.dataset_name, self.model, self.seed)
        self.g_model = deepcopy(self.model_0)
        self.kept_clients = [i for i in range(self.M)]
        self.P_kept = np.ones(self.M) / self.M
        self.n_aggregs = [self.T]
        self.best_models = [[deepcopy(self.model_0) for _ in range(self.M)]]
        self.branch = Branch()
        self.r, self.t = 0, 0
        empty = np.zeros((1, self.n_aggregs[0] + 1, self.M))
        self.acc = deepcopy(empty)
        self.loss = deepcopy(empty)
        self.metric = deepcopy(empty)

    def get_train(self, file_name: str):
        print("load global model ...")
        path_file = f"saved_exp_info/final_model/{file_name}.pth"
        self.g_model.load_state_dict(torch.load(path_file, map_location="cpu"))
        print("Done.")

        print("load best models ...")
        path_base = f"saved_exp_info/final_model/{file_name}"
        for i in range(self.M):
            self.best_models[0][i].load_state_dict(torch.load(path_base + f"_{0}_{i}.pth", map_location="cpu"))
        print("Done.")

        def hist_load(metric: str) -> np.array:
            with open(f"saved_exp_info/{metric}/{file_name}.pkl", "rb") as file_content:
                hist = pickle.load(file_content)
            return hist

        print("load acc, loss, metric history ...")
        hist_acc = hist_load("acc")
        self.acc[:, :hist_acc.shape[1]] = hist_acc
        hist_loss = hist_load("loss")
        self.loss[:, :hist_loss.shape[1]] = hist_loss
        hist_metric = hist_load("metric")
        self.metric[:, :hist_metric.shape[1]] = hist_metric
        self.train_length = hist_loss.shape[1]
        print("Done.")

        self.r, self.t = 0, self.T

    def hists_increase(self):
        empty = np.zeros((1, self.n_aggregs[-1] + 1, self.M))
        self.acc = np.concatenate((self.acc, empty), axis=0)
        self.loss = np.concatenate((self.loss, empty), axis=0)
        self.metric = np.concatenate((self.metric, empty), axis=0)

    def get_ref_vec(self, W_r):
        remaining_clients = [i for i in range(self.M) if i not in W_r]
        if not remaining_clients:
            print("警告：没有剩余客户端，使用全局模型作为参考模型。")
            return self.g_model.state_dict()

        ref_state_dict = {}
        for key in self.g_model.state_dict().keys():
            ref_state_dict[key] = sum(self.best_models[0][i].state_dict()[key] for i in remaining_clients) / len(
                remaining_clients)
        return ref_state_dict


    def forget_Wr(self, W_r: list[int]):
        print(f"forgetting {W_r}")

        if self.r == 0 and self.t == 0:
            print("no client to forget")
            return
        elif len(W_r) == 0:
            raise Exception("Forgetting requests cannot be empty")
        elif len(W_r) > 0:
            self.r += 1
            self.t = 0
            self.n_aggregs.append(self.T)
            self.hists_increase()

        for e in W_r:
            self.forget_clients.add(e)  # 记录遗忘客户端
            self.kept_clients.remove(e)
            self.P_kept[e] = 0.0
        self.P_kept /= sum(self.P_kept)

        print("forgetting with", self.unlearn_scheme)

        self.branch.update(self.metric, self.r, W_r, self.psi_star)
        print("branch", self.branch())
        zeta_r, T_r = self.branch()[-1]
        client = W_r[np.argmax(self.metric[zeta_r, T_r, W_r])]
        self.g_model = deepcopy(self.best_models[zeta_r][client])
        ul_state_dicts = {}
        ul_state_dicts[client] = copy.deepcopy(self.model_ul.state_dict())
        self.model_ul.load_state_dict(ul_state_dicts[client])
        global_state_dict = copy.deepcopy(self.g_model.state_dict())
        self.model_ul.load_state_dict(global_state_dict, strict=False)
        temp_state_dict = copy.deepcopy(self.model_ul.state_dict())
        temp_state_dict['classifier_ul.weight'] = (1 - gamma) * temp_state_dict['classifier.weight'] + gamma * \
                                                      temp_state_dict['classifier_ul.weight']
        temp_state_dict['classifier_ul.bias'] = (1 - gamma) * temp_state_dict['classifier.bias'] + gamma * \
                                                    temp_state_dict['classifier_ul.bias']
        self.model_ul.load_state_dict(temp_state_dict)
        noise_model(self.model_ul, self.sigma, self.device)
        self.best_models.append([deepcopy(self.model_ul) for _ in range(self.M)])

    def sample_clients(self):
        working_clients = random.sample(self.kept_clients, min(self.n_sampled, len(self.kept_clients)))
        return working_clients

    def aggregation(self, received_models: list[Module], model: Module) -> Module:
        P = [1 / len(received_models)] * len(received_models)
        if self.DP:
            vec_g = nn.utils.parameters_to_vector(deepcopy(model).parameters()).detach()
            for l, model_l in enumerate(received_models):
                vec_l = nn.utils.parameters_to_vector(deepcopy(model_l).parameters()).detach()
                alpha = max(torch.norm(vec_l - vec_g, p=2).cpu() / self.model_clip, 1.)
                P[l] /= alpha

        for w_k in model.parameters():
            w_k.data *= (1 - self.lr_g * np.sum(P))

        for p_i, rec_i in zip(P, received_models):
            for new_w_k, r_w_ik in zip(model.parameters(), rec_i.parameters()):
                new_w_k.data.add_(self.lr_g * p_i * r_w_ik)

        if self.DP:
            noise_model(self.g_model, self.sigma_DP, self.device)

        self.t += 1

    def compute_metric(self, working_clients: np.array, local_models: list[Module]):
        coef_regu = (1 - self.lr_l * self.lambd) ** self.n_SGD
        self.metric[self.r, self.t] = coef_regu * self.metric[self.r, self.t - 1]

        vec_g = nn.utils.parameters_to_vector(deepcopy(self.g_model).parameters()).detach()

        for l, model_l in zip(working_clients, local_models):
            vec_l = nn.utils.parameters_to_vector(deepcopy(model_l).parameters()).detach()
            coef_client_removal = self.lr_g * self.P_kept[l] / (1 - self.P_kept[l])
            self.metric[self.r, self.t, l] += coef_client_removal * torch.norm(vec_g - vec_l, p=2)


    def keep_best_model(self):
        for i in self.kept_clients:
            if self.metric[self.r, self.t, i] <= self.psi_star:
                self.best_models[self.r][i] = deepcopy(self.g_model)


    def loss_acc_global(self, clients: Clients):
        loss_acc = np.array(
            [client.loss_acc(self.g_model, self.loss_f, "train") for client in clients.clients]
        )

        self.loss[self.r, self.t] = loss_acc[:, 0]
        self.acc[self.r, self.t] = loss_acc[:, 1]

        loss = np.dot(self.P_kept, self.loss[self.r, self.t])
        var_loss = np.dot(self.P_kept, (loss - self.loss[self.r, self.t]) ** 2) ** 0.5

        accs = np.dot(self.P_kept, self.acc[self.r, self.t])
        var_acc = np.dot(self.P_kept, (accs - self.acc[self.r, self.t]) ** 2) ** 0.5

        print(
            f"====> n: {self.t} Train Loss: {loss:.3} +- {var_loss:.3} "
            f"Train Accuracy: {accs:.3} +- {var_acc:.3} %"
        )

        return loss, accs

    def compute_psi_bound(self, local_grads, working_clients):
        assert (len(self.policy) == 1)
        policy = self.policy[0]
        with torch.no_grad():
            N = len(working_clients)
            M = N - len(policy)
            psi_increment = torch.zeros_like(local_grads[0])
            for i, client in enumerate(working_clients):
                weight = 1 / N
                if client not in policy:
                    weight -= 1 / M
                psi_increment += weight * local_grads[i]
            return torch.norm(psi_increment)