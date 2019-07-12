# RL Technique

| Technique                 | Benefit                                   | Mentioned Key Algorithm                                      |
| ------------------------- | ----------------------------------------- | ------------------------------------------------------------ |
| Target network            | Stabilize the training process            | [DQN, 2015](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  |
| Memory buffer             | Breaking data relevance                   | [DQN, 2015](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  |
| KL-constrained update     | Optimize update step size                 | [TRPO, 2015](https://arxiv.org/pdf/1502.05477.pdf)           |
| Advantage function        | Stabilize learning                        | [A3C, 2015](https://arxiv.org/pdf/1602.01783.pdf)            |
| Importance sampling       | Data efficient                            | [PER,2016](https://arxiv.org/pdf/1511.05952.pdf)             |
| Entropy-regularized       | Better exploration                        | [Soft Q-Learning, 2018](https://arxiv.org/pdf/1704.06440.pdf) |
| Boltzmann policy          | Richer mathematical meaning               | [Soft Q-Learning, 2018](https://arxiv.org/pdf/1704.06440.pdf) |
| Target policy smoothing   | Avert Q-function incorrect sharp peak     | [TD3, 2018](https://arxiv.org/pdf/1802.09477.pdf)            |
| Clipped double-Q learning | Fend off overestimation in the Q-function | [TD3, 2018](https://arxiv.org/pdf/1802.09477.pdf)            |
| Reparameterize the policy | Lower variance estimate                   | [SAC, 2018](https://arxiv.org/pdf/1801.01290.pdf)            |

**PS: "Mentioned Key Algorithm" may not be the first algorithm that uses this technique, but makes a detailed explanation**