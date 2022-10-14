# HRL-habitat-lab

## Installing Habitat-Sim and Downloading data

先去habitat-challenge的[rearrangement分支下](https://github.com/facebookresearch/habitat-challenge/tree/rearrangement-challenge-2022)安装**habitat**环境、**habitat-sim** package以及**下载数据**

## Install Habitat-Lab

然后clone当前仓库, 进入HRL-habitat-lab目录,安装环境**habitat**

```
git clone https://github.com/kevin-xuan/HRL-habitat-lab.git
cd HRL-habitat-lab
pip install -r requirements.txt
python setup.py develop --all
```

## Train HRL ppo
```
cd scripts
sbatch hrl-ppo.sh
```
