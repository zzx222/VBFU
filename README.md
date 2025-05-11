## Verifiable Batch Federated Unlearning


### Download the Dependencies

This work is done using Python 3.9 and PyTorch 1.11.0.


### Abstract

As regulations around the world gradually regulate Artificial Intelligence (AI), Machine Unlearning (MU) reflects the importance of AI privacy protection. However, as an important part of AI architecture, existing works in Federated Unlearning (FU) face two major challenges: 1) the inability to efficiently handle large-scale unlearning requests and 2) the reliance on intrusive verification mechanisms, which may introduce additional risks to reduce global model performance and privacy leakage through intrusive verification. To tackle the aforementioned challenges, we propose \textit{Verifiable Batch Federated Unlearning} (VBFU), which initially employs a model evaluation module (MEval) to assess the sensitivity of the contribution of each client and snapshot the global model with the highest sensitivity. For verification purposes, the server marks the samples with the highest loss and variance for clients with unlearning requests. After marking, the memory erasure module (MErase) retrieves the stored snapshot from the MEval and builds up a Noise Augmented Fusion model (NAF) to complete the unlearning request. The final verification procedure involves checking the model variance in loss of marked samples before and after the MErase. Experiments were conducted on the CIFAR10, CIFAR100, MNIST, and FashionMNIST datasets, and the results demonstrate that VBFU not only outperforms state-of-the-art unlearning methods in handling large-scale unlearning requests but also efficiently achieves verification while maintaining model accuracy and privacy.

