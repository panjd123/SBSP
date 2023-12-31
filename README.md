# Ship Berth Scheduling Problem

RUC "Operations Research: Modeling and Algorithms 2023" Project

<img src="result/ga/plot-160.png" width=50%>

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

Reproduce the results in the report:

```
cd ./src
bash ./run-var.sh
make
bash ./run-ga.sh
```

> Note: The `run-var.sh` script will use "continue" mode to find the best solution. If you need to start from scratch, delete the `-c` parameter.
