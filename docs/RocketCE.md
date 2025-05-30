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

## Attribution

Much of this documentation is based on the articles below—credit goes to the original authors for their work.  
[Sizing and configuring an LPAR for AI workloads](https://community.ibm.com/community/user/blogs/sebastian-lehrig/2024/03/26/sizing-for-ai)  
[Install and use RocketCE in a Linux LPAR](https://community.ibm.com/community/user/blogs/sebastian-lehrig/2024/02/08/rocketce)
