# **Reinforcement Learning-based Relay Selection for Cooperative WSNs in the Presence of Bursty Impulsive Noise**

This repository is an accompaniment to the paper titled "RL-based Relay Selection for Cooperative WSNs in the Presence of Bursty Impulsive Noise," accepted for publication at IEEE Wireless Communications and Networking Conference (WCNC) 2024. Authored by Hazem Barka, Md Sahabul Alam, Georges Kaddoum, Minh Au, and Basile L. Agba. The paper presents innovative research on optimizing relay selection in Wireless Sensor Networks (WSN) through Reinforcement Learning (RL), particularly focusing on environments with bursty impulsive noise.

## Overview
The repository contains a collection of Python scripts and Jupyter notebooks that underpin the research presented in the paper. These files are essential for simulating WSNs, evaluating different relay selection strategies, and analyzing the impact of various noise models on network performance.

## Files Description

1. **`channels.py`**: Implements channel models, including AWGN and Rayleigh fading, used in the WSN simulations.

2. **`errorRates.py`**: Functions for calculating symbol error rates under different modulation schemes and channel conditions.

3. **`modem.py`**: Scripts for various modulation techniques such as PSK, QAM, and PAM.

4. **`TSMG noise Effect  Fast fading.ipynb`** and **`TSMG noise Effect slow fading.ipynb`**: These notebooks analyze the impact of TSMG noise in fast and slow fading scenarios, respectively.
   
5. **`Proposed max-min criterion.ipynb`**: Explores the performance of a proposed max-min relay selection criterion in WSNs.

6. **`RL - REINFORCE AGENT, reward maximization.ipynb`**: Implements a reinforcement learning agent to maximize rewards in relay selection within WSNs.
   
7. **`plotting everything.ipynb`**: Jupyter notebook for visualizing aspects of the WSN simulation, including error rates and network performance metrics for all methods.

## Contributing
We welcome contributions that can further enhance the research or extend the capabilities of the simulation environment.

## Citation

If you find this research or the repository helpful, please consider citing our paper:

```bibtex
@inproceedings{barka2024rlbased,
  title={RL-based Relay Selection for Cooperative WSNs in the Presence of Bursty Impulsive Noise},
  author={Barka, Hazem and Alam, Md Sahabul and Kaddoum, Georges and Au, Minh and Agba, Basile L.},
  booktitle={IEEE Wireless Communications and Networking Conference (WCNC)},
  year={2024}
}
```

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.
