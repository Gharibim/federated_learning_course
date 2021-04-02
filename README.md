# federated_learning_course
The materials for the Federated Learning Course Using [PyTorch](https://pytorch.org/) and [PySyft](https://github.com/OpenMined/PySyft) </br>
[Federated Learning course on Udemy](https://www.udemy.com/course/federated_learning)

[![plot](./fl.png)](https://www.udemy.com/course/federated_learning)


</br></br>
**What you will learn?**
* Introduction to Deep Learning and Neural Networks
* Introduction to Federated Learning
* Build Neural Networks from scratch using PyTorch
* Load your datasets in IID, non-IID, and non-IID unbalanced settings
* Introduction to PySyft
* Federated Learning techniques
  * FedAvg
  * FedSGD
  * FedProx
  * FedDANE
* Build your custom optimizer using PyTorch
* Introduction to Differential Privacy
* Implement FedAvg using Differential Privacy
* Federated Learning on cloud
* Implement FedAvg on cloud

</br></br>
You do not need any prerequisites!

</br></br>
PySyft version used in the course is `0.2.9` </br>
PySyft version used on the cloud is `0.3.0` </br>
You do not have to worry about torch version or any other library. All of these libraries are attached to PySyft and will be downloaded using a single command `pip install syft==<version>`</br>
When you are working on Differential Privacy and you want to use `Opacus` then you can install it using `pip install opacus`. However, `Opacus` is not compatible with torch `0.4` that comes with `PySyft`. To fix this, you need Torch `1.6` or above. To update your torch use `pip install torch==1.6`</br>
**Recommended**
Use three different anaconda envs:
* `Syft 0.2.9` with the default torch that comes with it. Use this env for ALL the local experiments.
* `Syft 0.2.9` with `torch 1.6` When you are on chapter 5 and you are running experiements using Opacus.
* `Syft 0.3.0` for all the cloud experiements.


</br></br>
**References**
* [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/pdf/1602.05629.pdf)
* [Federated Optimization in Heterogeneous Networks](https://arxiv.org/pdf/1812.06127.pdf)
* [FedDANE: A Federated Newton-Type Method](https://arxiv.org/pdf/2001.01920.pdf)
* [Calibrating Noise to Sensitivity in Private Data Analysis](https://people.csail.mit.edu/asmith/PS/sensitivity-tcc-final.pdf)
* [PyTorch](https://pytorch.org/)
* [PySyft](https://github.com/OpenMined/PySyft)
* [Opacus](https://opacus.ai/)
* [Google Federated Learning Blog](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
