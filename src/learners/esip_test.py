import copy

import numpy as np
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
from ER.PER.prioritized_memory import PER_Memory
import torch.nn as nn


#将正态分布门更新为绝对值计算的标准差和均值，在绝对值超过门时分享他的原值

class Esip:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.device = args.device
        self.params = list(mac.parameters())
        self.lam = args.lam
        self.alpha = args.alpha
        self.ind = args.ind
        self.mix = args.mix
        self.expl = args.expl
        self.dis = args.dis
        self.goal = args.goal
        #self.device = args.device #或许重复定义

        # 记录最后一次更新的时间
        self.last_target_update_episode = 0 #target模型的最后更新轮次
        self.last_gate_update_episode = 0 #经验共享和经验接收门的最后更新轮次
        # 选择mixer
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.target_mixer = copy.deepcopy(self.mixer)
        # 改动，分开做了mix和q的优化器与参数变量
        self.mixer_params = list(self.mixer.parameters())
        self.q_params = list(mac.parameters())
        self.mixer_optimiser = RMSprop(params=self.mixer_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.q_optimiser = RMSprop(params=self.q_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # 深拷贝有点浪费（如重复动作选择器），但对任何 MAC 都适用
        self.target_mac = copy.deepcopy(mac)
        # 记录数据的间隔轮数
        self.log_stats_t = -self.args.learner_log_interval - 1
        #计算distence的网络
        self.distance = nn.Sequential(
            nn.Linear(self.mac.scheme1['obs']['vshape'], 128),
            nn.ReLU(),
            nn.Linear(128, args.n_actions)
        ).to(device=self.device)
        #记录分布门对应信息
        self.q_mean=0 #智能体经验tderror分布的均值
        self.q_std=0 #智能体经验tderror分布的方差
        self.q_down_value=0 #智能体经验tderror分布的均值-方差
        self.q_up_value=0 #智能体经验tderror分布的均值+方差
        self.gate_warm_up_episode = 80 #在前多少轮固定更新一次正态分布门
        self.gate_if_reset = True #是否重新更新一次正态分布门（仅用于初始阶段）

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 获取相关变量
        rewards = batch["reward"][:, :-1]
        reward = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        #q_errors = batch["q_error"][:, :-1] #多余的变量
        observation = batch["obs"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        indi_terminated = batch["indi_terminated"][:, :-1].float()

        # 将经验导入到agent的网络计算Q函数
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # 随时间串联输出。 根据obs选择 动作 实际动作

        #筛选经验选中动作的分数，顺便去掉轨迹数目维度

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        #ind_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3) #多余变量

        # 使用target模型计算经验的Q函数
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        target_ind_q = th.stack(target_mac_out[:-1], dim=1)  ## 将个体的target_Q函数值全部拷贝下来，并按照时间链接
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # 计算目标时，我们不需要第一个时间步的 Q 值估计值，去除后按照时间链接

        # 将无法执行动作的Q函数值做负极大值掩膜
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        target_ind_q[avail_actions[:, :-1] == 0] = -9999999  # Q values  ##########################

        # #计算target_Q值的最大值
        if self.args.double_q:
            # #获取最大化实时Q的行动（DOUBLE-Qlearning）
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999 #对模型输出中的不可行动作Q函数做掩膜
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]#求每个时间步的最大Q函数动作

            cur_max_act = mac_out_detach[:, :-1].max(dim=3, keepdim=True)[1]  #记录当前每一步的最大Q函数动作，包括第一步，备用

            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3) #获取target模型计算的1-N时间步最大Q函数动
            target_individual_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3) #多余变量
            target_ind_qvals = th.gather(target_ind_q, 3, cur_max_act).squeeze(3)  #从包括第一步在内的每步最大动作来筛选对应个体Q函数
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0] #获取target模型计算的1-N时间步最大Q函数动作的对应Q值
            target_individual_qvals = target_mac_out.max(dim=3)[0]

        # 求解全局Q函数值
        if self.mixer is not None:
            chosen_action_qvals_clone = chosen_action_qvals.clone().detach()
            chosen_action_qvals_clone.requires_grad = True
            target_max_qvals_clone = target_max_qvals.clone().detach()

            chosen_action_q_tot_vals = self.mixer(chosen_action_qvals_clone, batch["state"][:, :-1])#计算选中动作的全局Q函数
            target_max_q_tot_vals = self.target_mixer(target_max_qvals_clone, batch["state"][:, 1:])#计算target模型选中动作的全局Q函数
            goal_target_max_qvals = self.target_mixer(target_ind_qvals, batch["state"][:, :-1])#计算个体Q组成的全局Q函数

        #####################################################################################################
        #本部分计算全局内在奖励

        # 计算ind_tot，也就是每个智能体的子目标
        q_ind_tot_list = []#每个agent的子目标的清单，是MASER论文的变量
        for i in range(self.n_agents):
            target_qtot_per_agent = (goal_target_max_qvals / self.n_agents).squeeze()# 每个agent平均分到的全局Q函数贡献
            q_ind_tot_list.append(self.alpha * target_ind_qvals[:, :, i] + (1 - self.alpha) * target_qtot_per_agent)#计算每个agent的子目标

        q_ind_tot = th.stack(q_ind_tot_list, dim=2)#规整子目标清单格式，按第二维拼接

        ddqn_qval_up_idx = th.max(q_ind_tot, dim=1)[1]  # 找出 t=1~T-1的最大 Q 值（对全部episode）

        #explore_q_target = th.ones(target_ind_q.shape) / target_ind_q.shape[-1] #多余变量
        #explore_q_target = explore_q_target.to(device=self.device)

        # 计算double-DQN的更新清单和距离计算清单
        ddqn_up_list = [] #double-DQN的更新清单
        distance_list = [] #距离清单
        for i in range(batch.batch_size):
            ddqn_up_list_subset = []
            distance_subset = []
            explore_loss_subset = [] #探索部分被删除了，多余变量
            for j in range(self.n_agents):
                # For distance function

                cos = nn.CosineSimilarity(dim=-1, eps=1e-8) #余弦相似度计算对象
                #cos1 = nn.CosineSimilarity() 多余变量
                goal_q = target_ind_q[i, ddqn_qval_up_idx[i][j], j, :].repeat(target_ind_q.shape[1], 1)
                #a = cos(target_ind_q[i, :, j, :], goal_q) 多余变量
                #b = cos1(target_ind_q[i, :, j, :], goal_q) 多余变量
                similarity = 1 - cos(target_ind_q[i, :, j, :], goal_q) #通过target模型的个体Q函数值和子目标Q值来计算相似度
                dist_obs = self.distance(observation[i, :, j, :]) #计算观察的距离
                dist_og = self.distance(observation[i, ddqn_qval_up_idx[i][j], j, :]) #计算子目标对应观察的距离

                # 计算距离损失
                dist_loss = th.norm(dist_obs - dist_og.repeat(dist_obs.shape[0], 1), dim=-1) - similarity
                distance_loss = th.mean(dist_loss ** 2)

                # 将Q子目标对应的状态观察和episode[i]j号agent距离子目标的距离损失载入清单
                distance_subset.append(distance_loss)
                ddqn_up_list_subset.append(observation[i, ddqn_qval_up_idx[i][j], j, :])

            distance1 = th.stack(distance_subset)
            distance_list.append(distance1)

            ddqn_up1 = th.stack(ddqn_up_list_subset)
            ddqn_up_list.append(ddqn_up1)

        # 计算子目标距离损失函数
        distance_losses = th.stack(distance_list)
        mix_explore_distance_losses = self.dis * distance_losses #乘上系数

        # 按照double-DQN计算奖励
        ddqn_up = th.stack(ddqn_up_list)
        ddqn_up = ddqn_up.unsqueeze(dim=1)
        ddqn_up = ddqn_up.repeat(1, observation.shape[1], 1, 1)
        reward_ddqn_up = self.distance(observation) - self.distance(ddqn_up)

        # 计算内在奖励
        intrinsic_reward_list = []
        for i in range(self.n_agents):
            intrinsic_reward_list.append(
                -th.norm(reward_ddqn_up[:, :, i, :], dim=2).reshape(batch.batch_size, observation.shape[1]))
        #intrinsic_rewards_ind = th.stack(intrinsic_reward_list, dim=-1)  # (B,T,n_agents) 多余变量
        intrinsic_rewards = th.zeros(rewards.shape).to(device=self.device)
        for i in range(self.n_agents):
            intrinsic_rewards += -th.norm(reward_ddqn_up[:, :, i, :], dim=2).reshape(batch.batch_size,
                                                                                     observation.shape[1],
                                                                                     1) / self.n_agents
        rewards += self.lam * intrinsic_rewards

        ####################################################################
        # 计算TD-error
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_q_tot_vals  # (B,T,1)#计算和筛选targetQ函数值
        td_error = (chosen_action_q_tot_vals - targets.detach())  # (B,T,1)
        #td_error_1 = (chosen_action_q_tot_vals - reward - self.args.gamma * (1 - terminated) * target_max_q_tot_vals) 多余变量

        #掩膜tderror
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask

        #计算混合网络的损失函数
        mix_explore_distance_loss = mix_explore_distance_losses.mean()
        mixer_loss = (masked_td_error ** 2).sum() / mask.sum() + 0.001 * self.mix * mix_explore_distance_loss

        # 优化混合网络
        self.mixer_optimiser.zero_grad()
        chosen_action_qvals_clone.retain_grad()  # the grad of qi
        chosen_action_q_tot_vals.retain_grad()  # the grad of qtot
        mixer_loss.backward()

        #计算mixer梯度信息，作为计算智能体内在奖励的一部分
        grad_l_qtot = chosen_action_q_tot_vals.grad.repeat(1, 1, self.args.n_agents) + 1e-8
        grad_l_qi = chosen_action_qvals_clone.grad
        grad_qtot_qi = th.clamp(grad_l_qi / grad_l_qtot, min=-10, max=10)  # (B,T,n_agents)
        mixer_grad_norm = th.nn.utils.clip_grad_norm_(self.mixer_params, self.args.grad_norm_clip)
        self.mixer_optimiser.step()

        #计算智能体的tderror
        q_rewards = self.cal_indi_reward(grad_qtot_qi, td_error, chosen_action_qvals, target_max_qvals,
                                         indi_terminated)  # (B,T,n_agents)(td_error_1)
        q_rewards_clone = q_rewards.clone().detach()
        q_targets = q_rewards_clone + self.args.gamma * (1 - indi_terminated) * target_max_qvals  # (B,T,n_agents)计算，筛选智能体单独target的Q函数值
        q_td_error = (chosen_action_qvals - q_targets.detach())  # (B,T,n_agents) 计算智能体单独的TD-ERROR

        #做掩膜
        q_mask = batch["filled"][:, :-1].float().repeat(1, 1, self.args.n_agents)  # (B,T,n_agents)
        q_mask[:, 1:] = q_mask[:, 1:] * (1 - indi_terminated[:, :-1]) * (1 - terminated[:, :-1]).repeat(1, 1,
                                                                                                        self.args.n_agents)
        q_mask = q_mask.expand_as(q_td_error)
        q2_mask = th.cat((q_mask, q_mask), dim=0)
        masked_q_td_error = q_td_error * q_mask

        #按照正态分布计算tderror的上下限（均值+-标准差）
        #更新，新增了间隔更新gate模式
        # 更新，使用绝对值来计算门和过门
        q_td_error_abs = th.abs(masked_q_td_error)  #新增绝对值版本的tderror
        if self.args.Normal_distribution_update == "every epoch":
            self.gate_if_reset = False #刚开始初始化，并关闭信号以便记录机制运行
            down_value = th.zeros((masked_q_td_error.shape[0], self.n_agents))
            up_value = th.zeros((masked_q_td_error.shape[0], self.n_agents))
            self.q_mean, self.q_std, self.q_down_value, self.q_up_value = self.sum_and_sig(q_td_error_abs,
                                                                                           down_value, up_value)#改用绝对值计算
        elif self.args.Normal_distribution_update == "interval update":
            if self.gate_if_reset == True and episode_num <= self.gate_warm_up_episode:#如果刚开始，初始化一次分布门
                self.gate_if_reset = False
                down_value = th.zeros((masked_q_td_error.shape[0], self.n_agents))
                up_value = th.zeros((masked_q_td_error.shape[0], self.n_agents))
                self.q_mean, self.q_std, self.q_down_value, self.q_up_value = self.sum_and_sig(q_td_error_abs,
                                                                                               down_value,
                                                                                               up_value)  # 改用绝对值计算
            elif (episode_num - self.last_gate_update_episode) / self.args.gate_update_interval >= 1.0:#每隔一定轮数更新分布门，目前是和更新target一个轮次
                self.last_gate_update_episode = episode_num
                down_value = th.zeros((masked_q_td_error.shape[0], self.n_agents))
                up_value = th.zeros((masked_q_td_error.shape[0], self.n_agents))
                self.q_mean, self.q_std, self.q_down_value, self.q_up_value = self.sum_and_sig(q_td_error_abs,
                                                                                               down_value,
                                                                                               up_value)  # 改用绝对值计算
        else:
            print("ERROR:参数Normal_distribution_update设置错误")
            exit()


        #计算共享tderror的清单
        share_list = th.zeros((masked_q_td_error.shape[0], masked_q_td_error.shape[1], self.n_agents))
        for i in range(masked_q_td_error.shape[0]):
            for j in range(masked_q_td_error.shape[1]):
                for agent in range(self.n_agents):
                    if q_td_error_abs[i][j][agent] >= self.q_up_value[i][agent]: # 绝对值过门
                        share_list[i][j][agent] = masked_q_td_error[i][j][agent]

       #计算接受共享tderror的清单
        #print(share_list.detach().numpy())
        print("分享：",share_list.sum())
        jieshou_list = th.zeros(share_list.shape[0], masked_q_td_error.shape[1], self.n_agents)
        share_list_abs = th.abs(share_list) #更新，使用绝对值过门
        for i in range(masked_q_td_error.shape[0]):
            for j in range(masked_q_td_error.shape[1]):
                dim = 0
                for agent in range(self.n_agents):
                    if agent == dim:
                        jieshou_list[i][j][agent] = 0 #不接受自己发出的经验
                    else:
                        if share_list_abs[i][j][dim] >= self.q_up_value[i][agent]:
                            # 如果分享单子上记着的tderror在自己这边也在均值+-标准差之外，记录
                            jieshou_list[i][j][agent] = share_list[i][j][dim]
                dim = dim + 1
        jieshou_list = jieshou_list.to(self.device)
        #print(jieshou_list.detach().numpy())
        print("接收",jieshou_list.sum())
        # 计算智能体的loss
        # 链接接受清单的tderror和智能体原有的tderror
        masked_q_td_error = th.cat((masked_q_td_error, jieshou_list), dim=0)
        # 计算轨迹的PER优先级权重和选取比率
        q_selected_weight, selected_ratio = self.select_trajectory(masked_q_td_error.abs(), q2_mask, t_env)
        q_selected_weight = q_selected_weight.clone().detach()
        # 计算智能体的loss
        q_loss = (masked_q_td_error ** 2 * q_selected_weight).sum() / q2_mask.sum()

        # 对智能体网络做优化
        self.q_optimiser.zero_grad()
        q_loss.backward()
        q_grad_norm = th.nn.utils.clip_grad_norm_(self.q_params, self.args.grad_norm_clip)
        self.q_optimiser.step()

        # 定期更新target模型
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num


        #定期打印信息
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("selected_ratio", selected_ratio, t_env)
            self.logger.log_stat("mixer_loss", mixer_loss.item(), t_env)
            self.logger.log_stat("mixer_grad_norm", mixer_grad_norm, t_env)
            # mask_elems = mask.sum().item()
            # self.logger.log_stat("mixer_td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            # self.logger.log_stat("mixer_target_mean", (targets * mask).sum().item() / mask_elems, t_env)
            # self.logger.log_stat("q_loss", q_loss.item(), t_env)
            # self.logger.log_stat("q_grad_norm", q_grad_norm, t_env)
            # q_mask_elems = q2_mask.sum().item()
            # self.logger.log_stat("q_td_error_abs", (masked_q_td_error.abs().sum().item() / q_mask_elems), t_env)
            # self.logger.log_stat("q_q_taken_mean", (chosen_action_qvals * q_mask).sum().item() / (q_mask_elems), t_env)
            # self.logger.log_stat("mixer_target_mean", (q_targets * q_mask).sum().item() / (q_mask_elems), t_env)
            # self.logger.log_stat("reward_i_mean", (q_rewards * q_mask).sum().item() / (q_mask_elems), t_env)
            # self.logger.log_stat("q_selected_weight_mean", (q_selected_weight * q2_mask).sum().item() / (q_mask_elems),
            #                      t_env)

            self.log_stats_t = t_env
        #print("本轮运行完成")



    # 函数：计算智能体tderror的均值，标准差，均值+-标准差的结果
    # 输入：掩膜后的智能体tderror列表masked_q_td_error。待填写的上下限清单down_value，up_value
    # 输出：每个batch的均值each_batch_mean，标准差each_batch_std，下限down_value，上限up_value
    def sum_and_sig(self, masked_q_td_error, down_value, up_value):

        each_batch_mean = th.mean(masked_q_td_error, dim=1) #计算均值
        each_batch_std = th.std(masked_q_td_error, dim=1) #计算标准差

        for i in range(each_batch_mean.shape[0]):
            for j in range(self.n_agents):
                down_value[i][j] = each_batch_mean[i][j] - each_batch_std[i][j]
                up_value[i][j] = each_batch_mean[i][j] + each_batch_std[i][j]
        return each_batch_mean, each_batch_std, down_value, up_value  # torch.Size([32,3])

    # 函数：计算个体奖励
    # 输入：grad_qtot_qi, mixer_td_error, qi, target_qi, indi_terminated
    # 输出：个体内在奖励
    def cal_indi_reward(self, grad_qtot_qi, mixer_td_error, qi, target_qi, indi_terminated):

        grad_td = th.mul(grad_qtot_qi, mixer_td_error.repeat(1, 1, self.args.n_agents))  # (B,T,n_agents)
        reward_i = - grad_td + qi - self.args.gamma * (1 - indi_terminated) * target_qi
        return reward_i

    #函数：计算轨迹的选择优先级分数
    #输入：智能体的tderror矩阵tderror，掩膜mask，轮次t_env
    def select_trajectory(self, td_error, mask, t_env):
        # td_error (B, T, n_agents)
        if self.args.warm_up:#如果采用预热
            if t_env / self.args.t_max <= self.args.warm_up_ratio:#在预热期内固定逐步上升选择比率
                selected_ratio = t_env * (self.args.selected_ratio_end - self.args.selected_ratio_start) / (
                        self.args.t_max * self.args.warm_up_ratio) + self.args.selected_ratio_start
            else:#超过预热期后固定选择比率
                selected_ratio = self.args.selected_ratio_end
        else:#固定选择比率
            selected_ratio = self.args.selected_ratio

        #按照不同的方式计算优先级权重
        if self.args.selected == 'all':
            return th.ones_like(td_error).cuda(), selected_ratio
        elif self.args.selected == 'greedy':
            valid_num = mask.sum().item()
            selected_num = int(valid_num * selected_ratio)
            td_reshape = td_error.reshape(-1)
            sorted_td, _ = th.topk(td_reshape, selected_num)
            pivot = sorted_td[-1]
            weight = th.where(td_error >= pivot, th.ones_like(td_error), th.zeros_like(td_error))
            return weight, selected_ratio
        elif self.args.selected == 'greedy_weight':
            valid_num = mask.sum().item()
            selected_num = int(valid_num * selected_ratio)
            td_reshape = td_error.reshape(-1)
            sorted_td, _ = th.topk(td_reshape, selected_num)
            pivot = sorted_td[-1]
            weight = th.where(td_error >= pivot, td_error - pivot, th.zeros_like(td_error))
            norm_weight = weight / weight.max()
            return norm_weight, selected_ratio
        elif self.args.selected == 'PER_hard':
            memory_size = int(mask.sum().item())
            selected_num = int(memory_size * selected_ratio)
            return PER_Memory(self.args, td_error, mask).sample(selected_num), selected_ratio
        elif self.args.selected == 'PER_weight':
            memory_size = int(mask.sum().item())
            selected_num = int(memory_size * selected_ratio)
            return PER_Memory(self.args, td_error, mask).sample_weight(selected_num, t_env), selected_ratio

    def get_gate_information(self):
        return self.q_mean,self.q_std,self.q_up_value,self.q_down_value

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.q_optimiser.state_dict(), "{}/q_opt.th".format(path))
        th.save(self.mixer_optimiser.state_dict(), "{}/mixer_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.q_optimiser.load_state_dict(th.load("{}/q_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer_optimiser.load_state_dict(
            th.load("{}/mixer_opt.th".format(path), map_location=lambda storage, loc: storage))
