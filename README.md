# Ship Berth Scheduling Problem

RUC "Operations Research: Modeling and Algorithms 2023" Project

## Problem Description

[doc/运筹学大作业.pdf](doc/%E8%BF%90%E7%AD%B9%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A.pdf)

## Project Report

[doc/report.pdf](doc/report.pdf)

## How to Run

### Requirements

- gurobipy
- numpy
- pandas
- matplotlib
- tqdm

### Run

Simple test:

```
cd ./src
mkdir ./temp
python main.py --help
python main.py -d 20 -g 20 -o temp
```

Reproduce the results in the report:

```
cd ./src
bash ./run.sh
```

> Note: The bash script will use "continue" mode to find the best solution. If you need to start from scratch, delete the `-c` parameter.
