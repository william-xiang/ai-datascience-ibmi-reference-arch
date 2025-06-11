# Rocket Cognitive Environment (RocketCE)

RocketCE offers a POWER-optimized software stack for running AI workloads. It has builtin exploitation of the [AI acceleration of the Power chipset](./RocketCE/mma.md). The product aims to minimize the entry barrier for AI by natively using Python packages on Linux LPARs, without any container platform needed. Benefit from over 200 packages optimized for IBM Power and backed by enterprise support from IBM and Rocket.

**Why choose it?**

- Keep your entire AI lifecycle on premise
- Exploit AI acceleration in IBM's Power hardware
- No container orchestration platform (such as OpenShift or Kubernetes) needed
- Evaluate at no cost

**Resources**

- [Announcement](https://www.ibm.com/docs/en/announcements/rocketce-aiml-power-support?region=US)
- [Community forum](https://community.rocketsoftware.com/forums/power?CommunityKey=c7ece6e8-5a29-4a17-a2bc-68b65f89d29f)

## Sizing and configuration

AI/ML workloads are typically CPU and memory intensive, which require substantial computing resources. In addition to the amount of CPU and memory allocated to an LPAR, configuration also plays a critical role in performance optimization. These workloads can benefit from Non-Uniform Memory Access(NUMA) architecture, which allows CPUs to access local memory faster than memory located on other NUMA nodes.

The content below shows the recommended CPU, memory allocation and configurations which is based on the content from this [article](https://community.ibm.com/community/user/blogs/sebastian-lehrig/2024/03/26/sizing-for-ai). Please refer to it for more details.

### Optimal Core Configuration by System

Given NUMA, the optimal configuration for cores would be 12 or 15 core SCMs (E1080), a 24 core DCM (E1050/S1024/L1024) is the second best option, followed by a 20 core DCM (S1022/L1022), and eventually 8 core eSCMs.

|System            |Module                                         |Core per Chip|
|------------------|-----------------------------------------------|-------------|
|E1080             |12 or 15 core SCMs(both perform similarly well)|12 or 15     |
|E1050/S1024/L1024 |24 core DCMs                                   |12           |
|S1022/L1022       |20 core DCMs                                   |10           |
|S1022s            |8 core eSCMs                                   |8            |

### NUMA Setup

The performance of AI workloads can be significantly affected by NUMA affinity. To optimize memory bandwidth and maximize performance, consider the following practices:

1. Confirm the P10 module (e.g., a 2x12 core DCM means that there are 2 DCMs with 12/2=6 cores per chip).
2. Setup an LPAR that allocates the max. number of cores available on the chip (so if you have 12 cores on the socket with a DCM, allocate 6 dedicated cores to the LPAR). This LPAR then corresponds to a so-called "NUMA node" and can access local memory fast.
3. Configure the LPAR as dedicated (and not shared) via the HMC.
4. Enable Power Mode in HMC (for full frequency exploitation).
5. Set SMT to 2 (but eventually try experimenting with 4 and 8).
6. (Re)start the system, ensuring that the LPAR from step 2 is started first, followed by the other LPARs. VIO servers typically do not cause conflicts in this order. Starting the target LPAR first helps ensure that it receives cores from a single chip, improving performance.

It's also recommended to perform the memory optimizations using command `optmem` in HMC to optimize the placement of LPAR to maximize the processor-memory affinity. Here are the commands for it:

1. Get the actual score of affinity of the LPAR, the score is a number between 0 and 100, with 0 representing the worst affinity and 100 representing perfect affinity.

    ```bash
    lsmemopt -m <system_name> -o currscore -r lpar --filter lpar_names=<lpar_name>

    ```

2. Calculate and list the potential partition affinity score after the memory optimization operation.

    ```bash
    lsmemopt -m <system_name> -o calcscore -r lpar -p <lpar_name>
    ```

3. Start a Dynamic Platform Optimization operation that prioritizes the specified LPAR.

    ```bash
    optmem -m <system_name> -o start -t affinity --id <lpar_id>
    ```

    > [!NOTE]
    > When using command optmem, the affinity score of other LPARs on the same system could be either positively or negatively impacted by the optimization.

4. Check the progress of the optimization. When it's complete, reboot the LPAR to apply the changes.

    ```bash
    lsmemopt -m <system_name>
    ```

5. Check if memory optimization works using command `lscpu` or `numactl -H`. Ideally you have only 1 NUMA node - only NUMA node0 - with its assigned CPUs. Below is the result of my own LPAR, it shows that there is only one NUMA node after optimization, compared to two nodes prior to the optimization.

    ```bash
    > numactl -H
    available: 1 nodes (7)
    node 7 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
    node 7 size: 95948 MB
    node 7 free: 94391 MB
    node distances:
    node 7
    7:   10
    ```

### System Memory

Memory should be sufficient for your intended workloads. Working with large language models (LLMs) typically requires around 80 GB of memory for a model with 20B parameters. To accommodate such demands, I often size LPARs with 256 GB of memory. Additionally, to maximize memory bandwidth, it’s important to populate all available memory slots with DIMMs—using several smaller DIMMs is preferable to a single large one.

### Storage

For running demos and POCs, 1 TB of disk space is typically more than sufficient.

## Installation

### Requirements

- Power9, and later, technology-based servers, with or without GPU
- Red Hat Enterprise Linux 9.0, or later
- Packages and binaries from the RocketCE channel

### Installation Guide

The binaries and packages optimized for IBM Power architecture are available in [RocketCE](https://anaconda.org/rocketce) channel. It is recommended to use Mamba instead of Conda to manage and resolve the packages, as Mamba is typically faster and can significantly reduce installation times, especially in large environments.

1. Run the commands below to install micromamba  

    ```bash
    dnf install bzip2 libxcrypt-compat vim -y
    "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
    source ${HOME}/.bashrc
    ```

    An environment is a directory that contains a specific collection of Conda/Mamba packages that you have installed. You can use different environments for different projects or tasks. If you change one environment, your other environments are not affected. There is a default environment called `base` that include a Python installation and some core system libraries and dependencies of Conda/Mamba. It's recommended to avoid installing additional packages into the base environment. Additional packages needed for a new project should always be installed into a newly created Conda environment.

    The commands below are used to create new environment, activate or deactivate environments.

    ```bash
    micromamba create -n myenv python=3.10 numpy pandas
    micromamba activate myenv
    micromamba deactivate myenv
    ```

    Please refer to the [Mamba documentation](https://mamba.readthedocs.io/en/latest/index.html) for more details.

2. Configure micromamba to use RocketCE

    ```bash
    cat > ~/.condarc <<'EOF'
    # Conda configuration see https://conda.io/projects/conda/en/latest/configuration.html
    auto_update_conda: false
    show_channel_urls: true
    channel_priority: flexible
    channels:
    - rocketce
    - defaults
    EOF
    ```

    > [!WARNING]
    > Adding Anaconda's defaults channel to above configuration requires an Anaconda license if you use it in a commercial context.

3. Install packages

    Those Python packages optimized for Power can be installed using micromamba command. Make sure you create a new environment to install these packages when you start a new project.  
    The command below is an example of how to install Python, PyTorch, and other packages using micromamba.

    ```bash
    micromamba install --yes python=3.10 pytorch-cpu mamba conda pip
    ```

    For packages that are not available in `defaults` and `RocketCE` channels, you can try installing them from `conda-forge` channel.

    ```bash
    micromamba install --yes 'conda-forge::accelerate'
    ```

    > [!INFO]
    > the conda-forge channel includes community-build packages; whereas the defaults and rocketce channels provide enterprise-grade builds and support.

    For packages that are not available in any Conda channels, you can choose to use `pip` to install them. But try to minimize using `pip` where possible as to keep your Conda environment clean and well-managed. You can preconfigure pip to use pre-build Python wheels from a repository by Power champion `Marvin Gießing` who precompiled some useful wheels, which speeds up package installations.

    - Optional: configure pip with Marvin's repos (recommended for rapid testing):

    ```bash
    mkdir ~/.pip && \
    echo "[global]" >> ~/.pip/pip.conf && \
    echo "extra-index-url = https://repo.fury.io/mgiessing" >> ~/.pip/pip.conf
    ```

    - Install pre-requisites from conda channels (in this example, these are needed for librosa):

    ```bash
    micromamba install --yes 'conda-forge::msgpack-python' 'conda-forge::soxr-python'
    ```

    - Install packages:

    ```pip install --prefer-binary \
    "librosa" \
    "openai-whisper"
    ```

4. Use JupyterLab

    JupyterLab is included in the RocketCE and it's the latest web-based interactive development environment for notebooks, code, and data.

    - Install JupyterLab

    ```bash
    mamba install --yes jupyterlab
    ```

    - Start JupyterLab

    ```bash
    mkdir notebooks
    nohup jupyter lab --notebook-dir=${HOME}/notebooks --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.allow_origin='*' --NotebookApp.token='' --NotebookApp.password='' &
    ```

    - Access JupyterLab using the URL: `http://<server>:8888/lab`

    The command used above to start JupyterLab does not specify a user password and uses HTTP for client-server communication. If you want to set a password and configure TLS/SSL for secure access, please refer to [Jupyter documentation](https://jupyter-server.readthedocs.io/en/latest/operators/public-server.html#jupyter-public-server).

    > [!WARNING]
    > If the firewall on the server is running, make sure it's configured to allow connections from client machines to access the port specified in the command above.

## Sample applications

### Intrusion Detection
With on-prem inferecing, deploying an intrusion detection system based on AI ob the IBM i system is an easy endeavor. Here, we train and compare two AI frameworks for intrusion detection, one that uses a neural network based system, particularly an multi-layer feed forward perceptron (MLP) which uses supervised learning, versus an unsupervised approach that uses an autoencoder deep learning framework. Both frameworks are trained and tested on the NSL-KDD dataset, with balanced classes for normal and attack vectors, and 43 different features to describe each tuple. Both the train and test samples can be downloaded using:

```bash
wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt
wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt
```

#### Multilayer Perceptron/MLP

An MLP is a fully connected feed-forward neural network, with an architecture that consists of linear layers, activation functions, loss functions and optimizers. The output is computed based on weights in the linear layers on the forward pass, where losses are calculated depending on the prediction, and then a backward pass is used to compute loss function derivatives, and then updated with the optimizer. This process continues throughout the training phase for the final model.

Here the necessary libraries are imported for the model.


```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

The KDDTrain and KDDTest samples are downloaded.


```python
!wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt
```

    --2025-06-10 16:53:44--  https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.109.133, 185.199.110.133, 185.199.108.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.109.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 19109424 (18M) [text/plain]
    Saving to: ‘KDDTrain+.txt’
    
    KDDTrain+.txt       100%[===================>]  18.22M  37.4MB/s    in 0.5s    
    
    2025-06-10 16:53:46 (37.4 MB/s) - ‘KDDTrain+.txt’ saved [19109424/19109424]
    



```python
!wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt
```

    --2025-06-10 16:53:48--  https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.110.133, 185.199.109.133, 185.199.108.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.110.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 3441513 (3.3M) [text/plain]
    Saving to: ‘KDDTest+.txt’
    
    KDDTest+.txt        100%[===================>]   3.28M  13.1MB/s    in 0.3s    
    
    2025-06-10 16:53:49 (13.1 MB/s) - ‘KDDTest+.txt’ saved [3441513/3441513]
    


Column names for the features, label and difficulty from the KDD Dataset are enumerated, and the data loaded from the CSV


```python
cols = [f'feature_{i}' for i in range(41)] + ['label', 'difficulty']
df = pd.read_csv("KDDTrain+.txt", header=None, names=cols)
```

The data is encoded, labels specified (there are numberous attack labels in the KDD Dataset, they are all classified under one label), and the features normalized. The data is then split into train and evaluation splits.


```python
# Encode categorical
for col in ['feature_1', 'feature_2', 'feature_3']:
    df[col] = LabelEncoder().fit_transform(df[col])

# Binary label: 0 = normal, 1 = attack
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Normalize features
X = df.iloc[:, :-2].values
y = df['label'].values
X = StandardScaler().fit_transform(X)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

This is where the Dataset is preprocessed before feeding to the model, by creating tensors, and after that the IDS model architecture is defined, with 3 Linear layers and 2 Rectified Linear Unit (ReLU) layers in-between each.


```python
class IntrusionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(IntrusionDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(IntrusionDataset(X_val, y_val), batch_size=64)
```


```python
class SimpleIDSModel(nn.Module):
    def __init__(self, input_size):
        super(SimpleIDSModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # No sigmoid
        )
    def forward(self, x):
        return self.net(x)
```

Model is trained with the various hyperparameters such as the criterion (loss function) and the optimizer. Epochs are the number of times the model goes through the training loop. Decreasing loss for each epoch means the model is learning, and more epochs may lead to better performance at the cost of training resources.


```python
model = SimpleIDSModel(X.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    for i, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass (logits only)
        logits = model(X_batch).view(-1)
        loss = criterion(logits, y_batch)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # optional
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Output monitoring for first batch in each epoch
        if i == 0:
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                print(f"\nEpoch {epoch+1}")
                print("  Logits:", logits[:5].numpy())
                print("  Probabilities:", probs[:5].numpy())
                print("  Targets:", y_batch[:5].numpy())
    
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
```

    
    Epoch 1
      Logits: [-0.09146228 -0.17194918 -0.16620973 -0.1687215  -0.10673405]
      Probabilities: [0.47715032 0.4571183  0.45854297 0.4579194  0.47334176]
      Targets: [1. 0. 0. 0. 1.]
    Epoch 1 Loss: 351.0091
    
    Epoch 2
      Logits: [-1.992386   6.0422745 -4.584759  -3.7289672 -4.790593 ]
      Probabilities: [0.12000466 0.99762946 0.01010309 0.02345431 0.00823908]
      Targets: [1. 1. 0. 0. 0.]
    Epoch 2 Loss: 127.9977
    
    Epoch 3
      Logits: [ 6.911761   -0.96739316  4.5789156   4.548613    3.0381422 ]
      Probabilities: [0.999005   0.2754004  0.98983824 0.98952895 0.95426786]
      Targets: [1. 0. 1. 1. 1.]
    Epoch 3 Loss: 95.3207
    
    Epoch 4
      Logits: [-5.6969457 -4.7877383 -5.778388  -5.5837092  7.5258374]
      Probabilities: [0.00334497 0.00826244 0.00308416 0.00374452 0.99946135]
      Targets: [0. 0. 0. 0. 1.]
    Epoch 4 Loss: 72.6620
    
    Epoch 5
      Logits: [-6.0059843 13.65903   -5.0342255 -6.4327526 10.449551 ]
      Probabilities: [0.00245791 0.9999988  0.00646912 0.00160544 0.99997103]
      Targets: [0. 1. 0. 0. 1.]
    Epoch 5 Loss: 59.6291
    
    Epoch 6
      Logits: [ -5.6283956 -10.749986    8.462506   11.078273   -7.4283724]
      Probabilities: [3.5814645e-03 2.1445256e-05 9.9978882e-01 9.9998450e-01 5.9380097e-04]
      Targets: [0. 0. 1. 1. 0.]
    Epoch 6 Loss: 52.0563
    
    Epoch 7
      Logits: [-6.1015778 11.520893  11.554005  10.352732  -6.8339925]
      Probabilities: [0.00223433 0.9999901  0.99999034 0.99996805 0.00107539]
      Targets: [0. 1. 1. 1. 0.]
    Epoch 7 Loss: 47.3041
    
    Epoch 8
      Logits: [ 8.413296  -9.12116   -5.695891  -4.9592686 -9.389942 ]
      Probabilities: [9.9977821e-01 1.0931588e-04 3.3484926e-03 6.9691497e-03 8.3553306e-05]
      Targets: [1. 0. 0. 0. 0.]
    Epoch 8 Loss: 43.9304
    
    Epoch 9
      Logits: [ 7.702268  -9.752635  -9.433811   8.834523  -7.9970593]
      Probabilities: [9.9954838e-01 5.8137877e-05 7.9967424e-05 9.9985433e-01 3.3633737e-04]
      Targets: [1. 0. 0. 1. 0.]
    Epoch 9 Loss: 41.0687
    
    Epoch 10
      Logits: [ 9.179084 10.527233 21.593142  8.310046  8.564558]
      Probabilities: [0.9998969 0.9999732 1.        0.999754  0.9998093]
      Targets: [1. 1. 1. 1. 1.]
    Epoch 10 Loss: 38.8511
    
    Epoch 11
      Logits: [ -7.450081 -12.375889  12.016492   9.549978  13.666914]
      Probabilities: [5.8105681e-04 4.2190818e-06 9.9999392e-01 9.9992883e-01 9.9999881e-01]
      Targets: [0. 0. 1. 1. 1.]
    Epoch 11 Loss: 37.4427
    
    Epoch 12
      Logits: [  0.7380047  -9.400427   10.752464  -11.815881   -6.3663497]
      Probabilities: [6.7655933e-01 8.2681923e-05 9.9997866e-01 7.3862666e-06 1.7154726e-03]
      Targets: [1. 0. 1. 0. 0.]
    Epoch 12 Loss: 35.7199
    
    Epoch 13
      Logits: [-16.691273   8.068301  -8.301234  13.128547  -4.831968]
      Probabilities: [5.6373153e-08 9.9968684e-01 2.4814869e-04 9.9999797e-01 7.9077892e-03]
      Targets: [0. 1. 0. 1. 0.]
    Epoch 13 Loss: 34.7277
    
    Epoch 14
      Logits: [ -7.8866506  -7.002042  -14.27597     9.264672   14.00633  ]
      Probabilities: [3.7558482e-04 9.0919464e-04 6.3099299e-07 9.9990535e-01 9.9999917e-01]
      Targets: [0. 0. 0. 1. 1.]
    Epoch 14 Loss: 33.5672
    
    Epoch 15
      Logits: [  6.335084   -9.381743  -10.329523   -9.518926   -5.3321953]
      Probabilities: [9.9823016e-01 8.4241088e-05 3.2653592e-05 7.3443138e-05 4.8101977e-03]
      Targets: [1. 0. 0. 0. 0.]
    Epoch 15 Loss: 32.6898
    
    Epoch 16
      Logits: [-9.922427 12.140923  9.988167  9.508379 -9.399144]
      Probabilities: [4.9059523e-05 9.9999464e-01 9.9995410e-01 9.9992573e-01 8.2788043e-05]
      Targets: [0. 1. 1. 1. 0.]
    Epoch 16 Loss: 31.8510
    
    Epoch 17
      Logits: [ 12.138786   -8.219288   10.727863  -12.77737     7.9523435]
      Probabilities: [9.9999464e-01 2.6933430e-04 9.9997807e-01 2.8239519e-06 9.9964833e-01]
      Targets: [1. 0. 1. 0. 1.]
    Epoch 17 Loss: 31.2835
    
    Epoch 18
      Logits: [ 14.749927 -12.818765  12.938683 -11.881991  -7.97953 ]
      Probabilities: [9.9999964e-01 2.7094434e-06 9.9999762e-01 6.9137504e-06 3.4228317e-04]
      Targets: [1. 0. 1. 0. 0.]
    Epoch 18 Loss: 30.6485
    
    Epoch 19
      Logits: [-13.378277 -15.057227 -12.840031  -9.703639  12.201403]
      Probabilities: [1.5484156e-06 2.8888780e-07 2.6524328e-06 6.1057159e-05 9.9999499e-01]
      Targets: [0. 0. 0. 0. 1.]
    Epoch 19 Loss: 30.1179
    
    Epoch 20
      Logits: [ -9.442284  -10.361995    4.589671   -6.6981816  15.36125  ]
      Probabilities: [7.9292826e-05 3.1610336e-05 9.8994595e-01 1.2316334e-03 9.9999976e-01]
      Targets: [0. 0. 1. 0. 1.]
    Epoch 20 Loss: 29.2303


Here the model is evaluated, and scores are calculated. The model shows a 99$ accuracy for the training data. 


```python
model.eval()
y_preds, y_trues = [], []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        logits = model(X_batch).view(-1)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        
        y_preds.extend(preds.numpy())
        y_trues.extend(y_batch.int().numpy())

from sklearn.metrics import classification_report
print(classification_report(y_trues, y_preds))
```

                  precision    recall  f1-score   support
    
               0       0.99      0.99      0.99     13422
               1       0.99      0.99      0.99     11773
    
        accuracy                           0.99     25195
       macro avg       0.99      0.99      0.99     25195
    weighted avg       0.99      0.99      0.99     25195
    


Confusion matrix showing a pretty low number of false positives and negatives.


```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_trues, y_preds)
ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Attack']).plot()
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7ffec0979150>




    
<!-- ![png](MLP_IDS_files/MLP_IDS_17_1.png) -->
    


The model is saved for deployment.


```python
torch.save(model.state_dict(), "intrusion_detection_model.pth")
```

The model appears to run great on known data. However, the true evaluation for the model comes from its performance on unseen data, for which we use the KDDTest part of the dataset.


```python
cols = [f'feature_{i}' for i in range(41)] + ['label', 'difficulty']
df_test = pd.read_csv("KDDTest+.txt", header=None, names=cols)

# Encode categorical
for col in ['feature_1', 'feature_2', 'feature_3']:
    df_test[col] = LabelEncoder().fit_transform(df_test[col])

# Binary classification: normal = 0, attack = 1
df_test['label'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Separate features and labels
X_test_raw = df_test.iloc[:, :-2].values
y_test = df_test['label'].values

# Normalize using same scaler as training (refit StandardScaler if not saved)
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test_raw)

# Convert to PyTorch dataset
class IntrusionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

test_dataset = IntrusionDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64)

# -------------------------------
# 3. Load trained model
# -------------------------------
model = SimpleIDSModel(X_test.shape[1])
model.load_state_dict(torch.load("intrusion_detection_model.pth"))
model.eval()

# -------------------------------
# 4. Evaluate on test set
# -------------------------------
y_preds, y_trues = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        logits = model(X_batch).view(-1)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        y_preds.extend(preds.numpy())
        y_trues.extend(y_batch.int().numpy())

# -------------------------------
# 5. Print Evaluation Metrics
# -------------------------------
print(classification_report(y_trues, y_preds))

# Optional: Confusion Matrix
cm = confusion_matrix(y_trues, y_preds)
ConfusionMatrixDisplay(cm, display_labels=["Normal", "Attack"]).plot()
```

                  precision    recall  f1-score   support
    
               0       0.68      0.97      0.80      9711
               1       0.97      0.65      0.78     12833
    
        accuracy                           0.79     22544
       macro avg       0.82      0.81      0.79     22544
    weighted avg       0.84      0.79      0.79     22544
    





    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7ffeaeab1210>




    
<!-- ![png](MLP_IDS_files/MLP_IDS_21_2.png) -->
    


As we can see here, the model shows a 79% F1-score on unseen data, which is not bad, but still far from the 99% we saw for known data. This points to the possibility of the model overfitting on the training data, which means it's learning patterns are too specific to the dataset. 



## Attribution

Much of this documentation is based on the articles below—credit goes to the original authors for their work.  
[Sizing and configuring an LPAR for AI workloads](https://community.ibm.com/community/user/blogs/sebastian-lehrig/2024/03/26/sizing-for-ai)  
[Install and use RocketCE in a Linux LPAR](https://community.ibm.com/community/user/blogs/sebastian-lehrig/2024/02/08/rocketce)
