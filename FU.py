import argparse
import torch
from experiment import Experiment
from server import Server
from client import Clients
from VBFU_unlearning import vbfu_unlearning

parser = argparse.ArgumentParser(description="Experiment Parameters")

parser.add_argument(
    "--dataset_name",
    type=str,
    help="MNIST, MNIST-shard_X, FEMNIST, CIFAR10, CIFAR100",
    default="MNIST-shard_0",
)

parser.add_argument(
    "--unlearn_scheme",
    type=str,
    help="VBFU",
    default="train",
)

parser.add_argument(
    "--model",
    type=str,
    help="default, CNN",
    default="default",
)

parser.add_argument(
    "--P_type",
    type=str,
    help="client importance p_i: ratio, uniform(only), dirichlet_X",
    default="uniform",
)

parser.add_argument("--forgetting", type=str, help="forgetting requests policy", default="P0")

parser.add_argument("--T", type=int, help="training time", default=10)
parser.add_argument("--B", type=int, help="batch size", default=128)

parser.add_argument("--n_SGD", type=int, help="Number of SGD", default=1)
parser.add_argument("--n_SGD_cali", type=int, help="Number of SGD for calibration",
                    default=0)
parser.add_argument("--delta_t", type=int, help="Number of step to wait between two gradients estimations",
                    default=2)

parser.add_argument("--lr_g", type=float, help="global lr", default=1.0)
parser.add_argument("--lr_l", type=float, help="local lr", default=0.01)

parser.add_argument("--M", type=int, help="number of clients", default=10)
parser.add_argument("--n_sampled", type=int, help="number of sampled clients", default=10)

parser.add_argument("--clip", type=float, help="clip", default=0.5)
parser.add_argument("--p_rework", type=float, default=1.0)
parser.add_argument("--limit_train_iter", type=float, default=0)
parser.add_argument("--lambd", type=float, help="regularization term", default=0)

parser.add_argument("--epsilon", type=float, default=1)
parser.add_argument("--sigma", type=float, default=0.05)

parser.add_argument("--stop_acc", type=float, default=90)
parser.add_argument("--seed", type=int, help="seed", default=0)
parser.add_argument("--iter_min", type=int, default=50)
parser.add_argument("--compute_diff", type=bool,
                    help="Whether or not to compute the psi bound of SIFU. This will also save all the iterates of the model for the future computation of alpha with bounds_comparison.py",
                    default=False)

parser.add_argument(
    "--device",
    help="training with CPU or GPU",
    default=torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")),
)

if __name__ == "__main__":

    args = vars(parser.parse_args())

    exp = Experiment(args)
    print(exp.file_name)
    print(exp.file_name_train)

    server = Server(args)
    server.get_train(exp.file_name_train)

    clients = Clients(exp.dataset_name, exp.M, exp.B,exp.device)

    vbfu_unlearning(exp, server, clients)

    exp.hists_save(server, "acc", "loss", "metric")

    exp.save_global_model(server.g_model)

    print("EXPERIMENT IS FINISHED AND SAVED")
