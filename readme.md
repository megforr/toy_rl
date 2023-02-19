# RL Algorithms on Toy Environments

### Environments
* [Gym](https://gymnasium.farama.org/)
* [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
* [OpenSpiel](https://github.com/deepmind/open_spiel)
* [PyGame](https://www.pygame.org/news)
* [Stable Baselines Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)

### Algorithms
* Tabular Q-learning 
* Deep Q-Learning (DQN)
* Policy Gradients (PG)
  * REINFORCE 
    * On-policy Method 
    * Policy gradient: `$\nabla J ~= E[Q(s,a) \nabla log \pi(a|s)]$`
      * Scale of gradient is proportional to the value of the action taken: Q(s,a)
      * The gradient itself is equal to the gradient of the log probability of the action taken.
  * Proximal Policy Optimization (PPO) 
  * Deep Deterministic Policy Gradient (DDPG)
* Actor Critic
  * Advantage Actor Critic (A2C)
  * Soft Actor Critic (SAC)
* Multi-Arm Bandit (MAB)
  * Epsilon Greedy
  * Upper Confidence Bound-1
  * Thompson Sampling
  * Best Arm ID - Fixed Confidence
  * Best Arm ID - Fixed Budget 
* Contextual MAB (cMAB)
  * LinUCB

### Frameworks
  * Pytorch
    * [Pytorch Install](https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c)
    * [Pytorch help](https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c)