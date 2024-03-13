# AD4RL 

[//]: # (<div>)

[//]: # (    <img src="https://img.shields.io/badge/Ubuntu-E95420?style=flat-square&logo=Ubuntu&logoColor=white"/> )

[//]: # (    <img src="https://img.shields.io/github/languages/top/leexim/AD4RL"> )

[//]: # (    <img src="https://img.shields.io/github/languages/code-size/leexim/AD4RL"> )

[//]: # (    <img src="https://img.shields.io/github/license/LeeXim/AD4RL">)

[//]: # (    <img src="https://img.shields.io/github/v/release/LeeXim/AD4RL">)

[//]: # (  </div>)

[//]: # (  <br>)

## ICRA2024
#### AD4RL: Autonomous Driving based on Dataset-driven Deep Reinforcement Learning 
[[Webpage]](https://sites.google.com/view/ad4rl/%ED%99%88) 
[[Dataset]](https://drive.google.com/drive/folders/1OKUS8rqLZ_REk1SP8PyAVYE_MMwlSVvo?usp=sharing)


## 1. FLOW Framework
See https://flow-project.github.io/ for Detail information of this framework

### Installation
#### A. Anaconda with Python 3
1. Install prequisites: `sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6` 
2. Download the Anaconda installation file for Linux in [Anaconda](https://anaconda.com/), and unzip the file
3. Install Anaconda `bash ~/Downloads/Anaconda3-2023.03-1-Linux-x86_64.sh`
> **_NOTE:_**  we recommend you to running conda init '**yes**'.

#### B. FLOW installation
Following the below scripts in your terminal.
```
# Download FLOW github repo'.
git clone https://github.com/flow-project/flow.git
cd flow

# Create a conda env and install the FLOW
conda env create -f environment.yml
conda activate flow
python setup.py develop

# install flow on previoulsy created environment 
pip install -e .
```

###### B-1. SUMO installation
Install driving simulator (SUMO) 
```
bash scripts/setup_sumo_ubuntu1804.sh
which sumo
sumo --version
sumo-gui
```
Testing the connection between FLOW and SUMO
```
conda activate flow
python examples/simulate.py ring
```

###### B-2. Pytorch installation
Install torch: `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`
> **_NOTE:_**  Should install at least 1.6.0 version of pytorch (Recommend torch = 1.11.0 & cudatoolkit=10.2).\
> Check the [Pytorch Documents](https://pytorch.org/get-started/previous-versions/).

###### B-3. Ray RLlib installation
Install Ray: `pip install -U ray==0.8.7`
> **_NOTE:_**  Should install at least 0.8.6 version of Ray. (Recommend 0.8.7).

###### B-4. Python Library installation
```

```

## 2. AD4RL
### Repository
Clone this library: `git clone` (Will be updated)
### Dependencies Update
```
sh ./requirements/env_requirements.sh
```
### Driving Scenarios
We provide the three driving scenarios as following table.
- Click Driving Scenario, You can check the illustrative image about driving scenario
- Click exp_config, You can check the code about driving scenario

| Driving Scenario                                   | exp_config                                                    |
|----------------------------------------------------|---------------------------------------------------------------|
| [Cut-in](exp_configs%2FFig%2Fcutin_Part.pdf)       | [UnifiedRing](exp_configs%2Frl%2Fmultiagent%2FUnifiedRing.py) |
| [Lane Reduction](exp_configs%2FFig%2FLR_Part.pdf)  | [MA_4BL](exp_configs%2Frl%2Fmultiagent%2FMA_4BL.py)           |
| [Highway](exp_configs%2FFig%2FHigh_Part.pdf)       | [MA_5LC](exp_configs%2Frl%2Fmultiagent%2FMA_5LC.py)           |

### Dataset
You can access the AD4RL googledrive by clicking the title name of driving scenario. 

| [Cut-in](https://drive.google.com/drive/folders/1qWV2UugbpeENPEAxONr0DRGnh9fiXJJS?usp=sharing)    | [Lane Reduction](https://drive.google.com/drive/folders/1vUeZEKXBO2_TC9LdxkH6JS4rMootnMfb) | [Highway](https://drive.google.com/drive/folders/1ZPxCGrZGrIYkU6VsKUYGWeMSPGdZOWcH?usp=sharing) |
|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------| 
| cutin-expert                                                                                      | lanereduction-expert                                                                       | highway-expert                                                                                  | 
| cutin-medium                                                                                      | lanereduction-medium                                                                       | highway-medium                                                                                  | 
| cutin-random                                                                                      | lanereduction-random                                                                       | highway-random                                                                                  |
| cutin-expert-medium                                                                               | lanereduction-expert-medium                                                                | highway-expert-meidum                                                                           |
| cutin-expert-random                                                                               | lanereduction-expert-random                                                                | highway-expert-random                                                                           |
| cutin-humanlike                                                                                   | lanereduction-humanlike                                                                    | highway-humanlike                                                                               |

### Train
 ```
 python [algorithm] [exp_config] --dataset [dataset]
 ```
- **[algorithm]**: main_BC.py, main_BCQ.py, main_DDPGBC.py, main_EDAC.py, main_PLAS.py
- **[exp_config]**: UnifiedRing, MA_4BL, MA_5LC
- **[dataset]**: See the above table (e.g., cutin-expert, highway-NGSIM)
