# Deep Q-Learning

* Pytorch
  * [Cartpole](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html?highlight=parameter%20grad%20data)
    * Task: Move cart left and right to keep upright
    * Solved: ...
    * Temporal Difference (TD) Error from Bellman Equation
    * Huber Loss - MSE when error is small, MAE when error is large
    * Model = NN that predicts return from Q(s, left) and Q(s, right)
    * Decayed epsilon greedy action selection strategy
    * Example training [results]() on 600 episodes
  * [Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
    * Task: Land spaceship safely between flags 
    * Solved: ...
    * Same general structure as cartpole example above 