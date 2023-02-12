# Advantage Actor Critic

* Pytorch
  * [Cartpole](https://github.com/megforr/toy_rl/blob/main/A2C/cartpole_pytorch.py)
    * Advantage = returns (Gt) - values (Vt)
      * Subtracts out state dependent baselines to reduce gradient variance
    * Loss = sum of actor and critic loss
      * actor loss = - sum of the action log probabilities * advantage for each ts in eps
      * critic loss = Huber loss between Gt and Vt