import torch

import numpy as np
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Branch: # 用于管理FedVBU中的忘却分支
    # 初始化为一个包含节点的列表，每个节点是一个包含两个整数的列表，表示忘却索引和聚合轮次
    def __init__(self, branch: list[list[int, int]] = [[0, 0]]):
        self.branch = deepcopy(branch)

    # 当调用该实例时，它返回当前的分支状态
    def __call__(self):
        return self.branch


    # 根据提供的度量 metric、请求编号 r、忘却集合 W_r 和隐私预算 psi_star 来剪枝分支
    def cut(self, metric: np.array, r: int, W_r: list[int], psi_star: float):
        """
        Cut the branch to only keep nodes forgetting W_r.If they all forget W_r then add the previous flow
        保留只涉及忘却集合W_r的节点，如果所有节点都忘却了W_r，则添加前一个流程
        """

        # 如果没有客户端需要忘却，则不需要剪枝
        if len(W_r) == 0:
            return

        # 遍历分支中的每个节点  zeta_s忘却索引编号  T_s聚合轮次
        for i, (zeta_s, T_s) in enumerate(self.branch):
            # 如果节点对应的度量超过隐私预算，则剪枝到该节点之前
            if max(metric[zeta_s, T_s, W_r]) > psi_star:
                self.branch = self.branch[:i]
                # 在剪枝后的分支末尾添加一个特殊标记 -42，表示在该节点之前的所有节点都需要被忘却或删除
                self.branch.append([zeta_s, -42])
                return
        # 如果没有节点满足剪枝条件，则在末尾添加一个新的节点，标记为[r - 1, -42]
        self.branch.append([r - 1, -42])


    # 根据当前分支的状态和度量来确定下一个聚合轮次T
    def get_T(self, metric: np.array, zeta: int, W_r: list[int], psi_star: float) -> int:

        # 计算在特定忘却集合下，所有客户端的度量的最大值。这是为了找出最需要忘却操作的点。
        psi_Sr = np.max(metric[zeta, :, W_r], axis=0)

        # 找到所有超过隐私预算的度量索引。这些索引指向了需要进行忘却操作的聚合轮次。
        indices = np.where(psi_Sr > psi_star)[0]

        # 如果存在至少一个索引超过了隐私预算，选择这些索引中的最小值减1作为下一个聚合轮次T。减1是因为我们需要在达到这个度量值之前执行忘却操作。
        if len(indices) > 0:
            return np.min(indices) - 1
        # 如果没有超过隐私预算的度量，返回第一个度量大于0的最大索引作为聚合轮次
        else:
            return np.max(np.where(psi_Sr > 0)[0])


    # 更新分支状态，metric：当前度量 r：请求编号  W_r：忘却集合 psi_star:隐私预算
    def update(self, metric: np.array, r: int, W_r: list[int], psi_star: float):

        # 如果请求编号为零，则保持初始状态
        if r == 0:
            return

        #  调用cut方法来剪枝分支
        self.cut(metric, r, W_r, psi_star)

        # zeta_r：当前分支的最后一个节点
        zeta_r = self.branch[-1][0]
        # T_r：通过get_T获取下一个聚合轮次
        T_r = self.get_T(metric, zeta_r, W_r, psi_star)

        # 确保分支列表中不包含重复的节点
        if [zeta_r, T_r] == self.branch[-2]:
            # 如果存在重复节点，删除最后一个节点
            self.branch = self.branch[:-1]
        else:
            # 否则，更新最后一个节点的聚合轮次为T_r
            self.branch[-1][1] = T_r
