
# AT-DQN : Attention-Enhanced Exploration in Deep Reinforcement Learning

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

## Overview
This codebase implements a research work on Attention driven Deep Q-Network (AT-DQN) which offers an optimized exploration strategy as opposed to fixed heuristic and probabilistic techniques such as epsilon-greedy. This work produces a proof of concept for the proposed methodology by evaluating the algorithm on a suite of classic control benchmarks, demonstrating competitive convergence and rewards.

<img width="949" alt="Screenshot 2025-05-21 at 7 35 36 PM" src="https://github.com/user-attachments/assets/3655d48b-d48a-4999-b8a6-44057ef00ad9" />



## Directory structure

```
project_root/
├── Readme.md
├── Best_Models/
│   ├── Acrobot.pth   
│   ├── Cartpole.pth    
|   ├── lunalander.pth       
│   └── mountaincar.pth      
├── Evaluation_Scripts/
|   ├── eval.py #Single episode evaluation with render       
│   └── eval_100.py #Evaluation over n episodes    
│
└── Scripts/
│   ├── Acrobot.py
│   ├── Cartpole.py
|   ├── LunaLander.py
|   ├── MountainCar.py          
│   └── config.yaml
```




## Environments
<img width="506" alt="Screenshot 2025-05-21 at 7 40 53 PM" src="https://github.com/user-attachments/assets/36442427-c954-4a85-a5c6-5e2cf0593c10" />

## Reward Curves
<img width="472" alt="Screenshot 2025-05-21 at 7 41 37 PM" src="https://github.com/user-attachments/assets/a32934e4-b18a-4e8e-bc26-a28554ac1176" />


## Benchmarks


| **Policy**        | **Acrobot-v1**       | **CartPole-v1**      | **MountainCar-v0**     | **LunaLander-v2**      |
|-------------------|----------------------|-----------------------|------------------------|------------------------|
| DQN               | −81.81 ± 17.19       | 499.80 ± 0.14         | −110.89 ± 9.92         | −136.42 ± 9.92         |
| Prioritized-DQN   | −105.20 ± 14.74      | 498.70 ± 1.43         | No Benchmarks          | No Benchmarks          |
| DRQN              | −82.26 ± 14.34       | 132.50 ± 69.79        | No Benchmarks          | No Benchmarks          |
| REINFORCE         | −104.80 ± 14.51      | 500.00 ± 0.00         | No Benchmarks          | −146.74 ± 40.89        |
| PPO               | −77.22 ± 8.45        | 499.90 ± 0.33         | −110.42 ± 19.47        | −266.26 ± 12.92        |
| Rainbow           | −158.10 ± 55.48      | 478.30 ± 29.28        | −166.38 ± 27.94        | −218.30 ± 40.91        |
| SAC               | −121.00 ± 35.31      | 500.00 ± 0.00         | No Benchmarks          | No Benchmarks          |
| **AT-DQN (ours)** | **−77.39 ± 12.03**   | **500.00 ± 0.00**     | **−108.69 ± 6.82**     | **−210.31 ± 12.82**    |

<img width="478" alt="Screenshot 2025-05-21 at 7 44 14 PM" src="https://github.com/user-attachments/assets/39a5345d-229b-4c68-83f1-34b89331a200" />


**Funding:** H. Thangaraj conducted this research at the Council for Scientific and Industrial Research - Fourth Paradigm Institute, Bengaluru, India under the guidance of Nallana Mithun Babu. Additionally, this work was endorsed by Vellore Institute of Technology, Chennai, India with assistance from Dr. Abinaya S.



**Data Access Statement:** This study involves secondary analyses of existing RL environments, as well as the primary release of novel model architectures proposed by the authors.


## Appendix

**Note:** Please, when using any of the resources provided here, remember to cite this repository.

