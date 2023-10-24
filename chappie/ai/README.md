# AI

### Description

The following is a generalized reinforcement learning AI program. This AI is trained using a combination of reinforcement learning and evolutionary learning. This AI is based off of [DeepMind's](https://deepmind.com/) [MuZero](https://arxiv.org/abs/1911.08265) algorithm where it uses a Neural Network in combination with the Monte Carlo Tree Search Algorithm. However the Neural Network (NN) used in this AI differs from MuZero since in their papers they use a convolutional neural network (CNN) where with this AI a [Perceiver IO](https://arxiv.org/abs/2107.14795) based architecture is used. Self-supervised learning is also added to help the dynamic head learn a representation of the next action hidden state similarly to what was proposed in [EfficientZero](https://arxiv.org/abs/2111.00210).

### Reference

- https://arxiv.org/abs/1911.08265 (MuZero)
- https://arxiv.org/abs/2107.14795 (Perceiver IO)
- https://arxiv.org/abs/2111.00210 (EfficientZero)
