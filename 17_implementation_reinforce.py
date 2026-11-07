"""
Project 17: Mini NAS Controller
Algorithm: REINFORCE Policy Gradient with Exponential Moving Average Baseline
Target: RNN Controller proposing compact CNN topologies for CIFAR-10
Baseline: Uniform Random Architecture Search (Random NAS)
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# -------------------------------------------------------------
# 1. Search Space Specification
# -------------------------------------------------------------
SEARCH_SPACE = {
    "filters": [16, 32, 64],
    "kernel_size": [3, 5],
    "activation": ["relu", "leaky_relu", "gelu"],
    "residual": [0, 1]
}

NUM_LAYERS = 3
DECISION_KEYS = ["filters", "kernel_size", "activation", "residual"]
DECISION_CHOICES = [len(SEARCH_SPACE[k]) for k in DECISION_KEYS] * NUM_LAYERS
TOTAL_DECISIONS = len(DECISION_CHOICES)  # 4 * 3 = 12 decisions

def get_activation(act_name):
    if act_name == "relu":
        return nn.ReLU(inplace=True)
    elif act_name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_name == "gelu":
        return nn.GELU()
    return nn.ReLU(inplace=True)

# -------------------------------------------------------------
# 2. Dynamic Child CNN Generator
# -------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, act_name, use_res):
        super(ConvBlock, self).__init__()
        self.use_res = (use_res == 1) and (in_c == out_c)
        pad = k // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = get_activation(act_name)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        res = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        if self.use_res:
            out = out + res
        out = self.pool(out)
        return out

class DynamicChildCNN(nn.Module):
    def __init__(self, arch_config, num_classes=10):
        super(DynamicChildCNN, self).__init__()
        self.blocks = nn.ModuleList()
        in_c = 3
        for l in range(NUM_LAYERS):
            out_c = arch_config[f"layer_{l}_filters"]
            k = arch_config[f"layer_{l}_kernel_size"]
            act = arch_config[f"layer_{l}_activation"]
            res = arch_config[f"layer_{l}_residual"]
            self.blocks.append(ConvBlock(in_c, out_c, k, act, res))
            in_c = out_c

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_c, num_classes)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -------------------------------------------------------------
# 3. RNN Policy Controller (Autoregressive Generator)
# -------------------------------------------------------------
class RNNController(nn.Module):
    """
    Autoregressive RNN generating architectural tokens sequentially.
    """
    def __init__(self, hidden_dim=64):
        super(RNNController, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)

        # Output projection heads for each sequential decision
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_choices) for num_choices in DECISION_CHOICES
        ])
        self.embedding = nn.Embedding(max(DECISION_CHOICES) + 1, hidden_dim)

    def forward(self):
        hx = torch.zeros(1, self.hidden_dim, device=self.embedding.weight.device)
        cx = torch.zeros(1, self.hidden_dim, device=self.embedding.weight.device)

        actions = []
        log_probs = []
        entropies = []

        # Initial dummy input
        inp = torch.zeros(1, self.hidden_dim, device=hx.device)

        for step in range(TOTAL_DECISIONS):
            hx, cx = self.lstm(inp, (hx, cx))
            logits = self.heads[step](hx)
            dist = torch.distributions.Categorical(logits=logits)

            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            actions.append(action.item())
            log_probs.append(log_prob)
            entropies.append(entropy)

            # Embed chosen action as input for the next autoregressive step
            inp = self.embedding(action)

        return actions, torch.stack(log_probs).sum(), torch.stack(entropies).sum()

def decode_actions_to_arch(actions):
    """Translates the 12 sampled integers into structured architecture config."""
    config = {}
    idx = 0
    for l in range(NUM_LAYERS):
        config[f"layer_{l}_filters"] = SEARCH_SPACE["filters"][actions[idx]]
        idx += 1
        config[f"layer_{l}_kernel_size"] = SEARCH_SPACE["kernel_size"][actions[idx]]
        idx += 1
        config[f"layer_{l}_activation"] = SEARCH_SPACE["activation"][actions[idx]]
        idx += 1
        config[f"layer_{l}_residual"] = SEARCH_SPACE["residual"][actions[idx]]
        idx += 1
    return config

def sample_random_architecture():
    actions = [random.randint(0, num_choices - 1) for num_choices in DECISION_CHOICES]
    return actions, decode_actions_to_arch(actions)

# -------------------------------------------------------------
# 4. Fast Child Model Evaluator
# -------------------------------------------------------------
def train_and_eval_child(arch_config, train_loader, val_loader, device, child_epochs=3):
    model = DynamicChildCNN(arch_config).to(device)
    params = count_parameters(model)
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for ep in range(child_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Fast validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

    val_acc = correct / max(1, total)
    # Reward formulation: ValAcc (percentage) minus mild parameter footprint penalty
    reward = (val_acc * 100.0) - (0.05 * (params / 10000.0))
    return val_acc, params, reward

# -------------------------------------------------------------
# 5. Main NAS Search & Comparison Pipeline
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Mini NAS Controller with REINFORCE (Project 17)")
    parser.add_argument("--iterations", type=int, default=20, help="Total architecture search iterations")
    parser.add_argument("--child_epochs", type=int, default=3, help="Epochs to train each proposed child model")
    parser.add_argument("--subset", type=int, default=3000, help="CIFAR-10 training subset for rapid evaluation")
    parser.add_argument("--out_dir", type=str, default="17", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Project 17] Running Mini NAS on {device} | Iterations: {args.iterations} | Child Epochs: {args.child_epochs}")

    # Prepare CIFAR-10 DataLoaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
    val_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)

    train_subset = Subset(train_data, list(range(args.subset)))
    val_subset = Subset(val_data, list(range(min(1500, len(val_data)))))

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, pin_memory=True)

    # 1. Baseline: Random Architecture Search
    print("\n>>> Phase 1: Evaluating Random Architecture Search Baseline...")
    random_rewards = []
    random_val_accs = []
    random_best_acc = []
    curr_best_random = 0.0

    for i in range(1, args.iterations + 1):
        _, rand_arch = sample_random_architecture()
        val_acc, params, reward = train_and_eval_child(rand_arch, train_loader, val_loader, device, args.child_epochs)
        random_rewards.append(reward)
        random_val_accs.append(val_acc)
        curr_best_random = max(curr_best_random, val_acc)
        random_best_acc.append(curr_best_random)
        print(f"  [Random NAS {i:02d}/{args.iterations:02d}] ValAcc: {val_acc*100:5.2f}% | Params: {params:,} | Reward: {reward:6.2f} | Best: {curr_best_random*100:5.2f}%")

    # 2. RL Controller: REINFORCE with Baseline
    print("\n>>> Phase 2: Training RNN NAS Controller via REINFORCE...")
    controller = RNNController(hidden_dim=64).to(device)
    optimizer = optim.Adam(controller.parameters(), lr=0.01)

    controller_rewards = []
    controller_val_accs = []
    controller_best_acc = []
    curr_best_controller = 0.0
    best_discovered_arch = None

    baseline_b = 0.0  # Moving average baseline for variance reduction
    decay = 0.85

    for i in range(1, args.iterations + 1):
        actions, log_prob, entropy = controller()
        arch_config = decode_actions_to_arch(actions)

        val_acc, params, reward = train_and_eval_child(arch_config, train_loader, val_loader, device, args.child_epochs)

        if val_acc > curr_best_controller:
            curr_best_controller = val_acc
            best_discovered_arch = arch_config

        controller_rewards.append(reward)
        controller_val_accs.append(val_acc)
        controller_best_acc.append(curr_best_controller)

        # Baseline update
        if i == 1:
            baseline_b = reward
        else:
            baseline_b = decay * baseline_b + (1 - decay) * reward

        advantage = reward - baseline_b

        # REINFORCE policy loss + entropy bonus for exploration
        policy_loss = -log_prob * advantage - 0.01 * entropy

        optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        optimizer.step()

        print(f"  [RL Controller {i:02d}/{args.iterations:02d}] ValAcc: {val_acc*100:5.2f}% | Adv: {advantage:+6.2f} | Reward: {reward:6.2f} | Best: {curr_best_controller*100:5.2f}%")

    # Save Controller & Discovered Architecture
    torch.save(controller.state_dict(), os.path.join(args.out_dir, "controller_checkpoint.pt"))
    summary = {
        "best_discovered_arch": best_discovered_arch,
        "best_controller_val_acc": float(curr_best_controller),
        "best_random_val_acc": float(curr_best_random),
        "mean_controller_reward": float(np.mean(controller_rewards)),
        "mean_random_reward": float(np.mean(random_rewards))
    }
    with open(os.path.join(args.out_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 3. Generate Comparative Plots
    plot_path = os.path.join(args.out_dir, "learning_curves.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    iters = list(range(1, args.iterations + 1))

    # Left: Best Discovered Validation Accuracy Trajectory
    ax1.plot(iters, [a * 100 for a in controller_best_acc], label="REINFORCE Controller (Best)", color="#2ca02c", linewidth=2.5)
    ax1.plot(iters, [a * 100 for a in random_best_acc], label="Random Search Baseline (Best)", color="#d62728", linestyle="--", linewidth=2.2)
    ax1.set_title("Best Found Architecture Accuracy over Search Budget", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Search Iteration (Proposed Architectures)", fontsize=11)
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=10)

    # Right: Reward Progression (Smoothed Moving Average)
    window = 3
    smooth_ctrl = np.convolve(controller_rewards, np.ones(window)/window, mode='valid')
    smooth_rand = np.convolve(random_rewards, np.ones(window)/window, mode='valid')
    smooth_iters = list(range(window, args.iterations + 1))

    ax2.plot(smooth_iters, smooth_ctrl, label="REINFORCE Controller (Smoothed)", color="#2ca02c", linewidth=2.2)
    ax2.plot(smooth_iters, smooth_rand, label="Random Search Baseline (Smoothed)", color="#d62728", linestyle="--", linewidth=2.0)
    ax2.set_title("Reward Signal Progression (Smoothed EMA)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Search Iteration", fontsize=11)
    ax2.set_ylabel("Reward Value", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"\n[Project 17] Successfully saved Mini NAS curves plot to {plot_path}")

if __name__ == "__main__":
    main()

