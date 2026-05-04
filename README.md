# Reinforcement Learning for Robotics: Study and Research Notes

## 1. Introduction
This repository documents:

- A curated collection of learning resources for Reinforcement Learning (RL), including textbooks, courses, and papers  
- Structured notes and summaries of key RL concepts and algorithms (e.g., PPO, SAC, hierarchical RL)  
- Personal insights and reflections developed during the learning process  

The main objectives are:

- To organize high-quality RL learning materials into a clear and structured roadmap  
- To deepen understanding through summarization and critical reflection  
- To build a long-term knowledge base for reinforcement learning  


---

## 2. Learning Roadmap

### 2.1 Fundamentals

Mathematical foundation of reinforcement learning

Covers reinforcement learning from fundamental concepts to modern algorithms in a structured and mathematically grounded way. 

The content is organized into two parts: foundational tools (MDP, Bellman equations, dynamic programming) and core algorithms (Monte Carlo, TD learning, policy gradient, actor-critic). 

The chapters are highly connected and designed to be studied sequentially, making it a solid resource for building a systematic understanding of RL.

Note: This resource focuses primarily on theoretical foundations and classical RL methods, and does not cover more recent deep RL algorithms such as DDPG, TD3, PPO, or SAC.

Book: https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning

Video: https://www.youtube.com/watch?v=zJHtM5dN69g&list=PLEhdbSEZZbDaFWPX4gehhwB9vJZJ1DNm8



UC Berkeley CS285: Deep Reinforcement Learning  

A comprehensive course covering modern deep reinforcement learning, including both fundamental algorithms and advanced research topics. 

The course covers key methods such as policy gradients, actor-critic, Q-learning, as well as model-based RL, imitation learning, and offline RL. It also includes topics like exploration, transfer learning, and meta-learning, reflecting current research trends. 

The course combines theoretical understanding with practical implementation through programming assignments and a final research-oriented project, making it a strong resource for learning both concepts and applications. 

Note: This course assumes prior knowledge of machine learning and basic reinforcement learning, and may be challenging for beginners without sufficient background. 

Course: https://rail.eecs.berkeley.edu/deeprlcourse/



Deepmind: https://www.youtube.com/watch?v=TCCjZe0y4Qc&list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm&index=1

Reinforcement Learning: An Introduction: http://incompleteideas.net/book/the-book-2nd.html



---

### 2.2 Deep Reinforcement Learning

Deep Reinforcement Learning extends classical reinforcement learning by incorporating deep neural networks as function approximators, enabling agents to learn directly from high-dimensional inputs such as images and continuous state spaces.

DQN: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)

DDPG: [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

PPO: Proximal Policy Optimization Algorithms https://arxiv.org/abs/1707.06347

TD3: Addressing Function Approximation Error in Actor-Critic Methods https://arxiv.org/abs/1802.09477

SAC: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor https://arxiv.org/abs/1801.01290



### 2.3 New Research Topics

Recent advances in reinforcement learning have shifted from designing entirely new algorithms (e.g., PPO, SAC) to improving sample efficiency and data utilization.

Key research directions include:

- **Offline Reinforcement Learning**  
  Learning policies from fixed datasets without environment interaction, making RL applicable to safety-critical and data-rich domains  

  CQL: [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779)
  
  BCQ: [Off-Policy Deep Reinforcement Learning without Exploration](https://arxiv.org/abs/1812.02900)
  
  BEAR: [Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction](https://arxiv.org/abs/1906.00949)
  
  
  
- **Model-Based RL and World Models**  
  Learning environment dynamics to enable planning and improve sample efficiency, often combined with representation learning  

  PlaNet: [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)
  
  Dreamer: [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)
  
  


---

## 3. RL Tools and Frameworks

This section focuses on tools and libraries specifically designed for reinforcement learning.

### 3.1 RL Libraries

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3): Standard implementations of popular RL algorithms (PPO, SAC, DQN), widely used for benchmarking and prototyping  
- [RLlib (Ray RLlib)](https://docs.ray.io/en/master/rllib/index.html): Scalable RL library supporting distributed training and multi-agent systems  
- [CleanRL](https://github.com/vwxyzjn/cleanrl): Minimal and transparent implementations of RL algorithms, useful for understanding and research  


---

### 3.2 Environments and Benchmarks

- [Gym / Gymnasium](https://gymnasium.farama.org/index.html#): Standardized interface for RL environments  
  
- [MuJoCo](https://mujoco.org/): Continuous control benchmark environments  
  
- [DeepMind Control Suite](https://github.com/google-deepmind/dm_control): Physics-based control tasks for RL research  
  
- [Atari Learning Environment](https://ale.farama.org/index.html):Classic benchmark for evaluating RL algorithms  
