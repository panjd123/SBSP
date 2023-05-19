import pandas as pd
import os.path as osp
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, Normalize
import heapq
from tqdm import tqdm
import signal

parser = argparse.ArgumentParser()
parser.add_argument("--ship_dataset", type=str, default="tiny", required=False)
arg = parser.parse_args()

DATA_DIR = "data/"
PORT_DATA_FILE = osp.join(DATA_DIR, "ports.txt")
PORT_TINY_DATA_FILE = osp.join(DATA_DIR, "ports-tiny.txt")
SHIP_DATA_FILES = {
    "20": osp.join(DATA_DIR, "ships20.txt"),
    "40": osp.join(DATA_DIR, "ships40.txt"),
    "80": osp.join(DATA_DIR, "ships80.txt"),
    "160": osp.join(DATA_DIR, "ships160.txt"),
}
SHIP_TINY_DATA_FILE = osp.join(DATA_DIR, "ships-tiny.txt")


def calD(t, D_0=0, a=2, T=1440):
    # return D_0 + a * np.sin(2 * np.pi * t / T)
    return D_0 + a * np.sin(2 * np.pi * t / T)


_port_free = [[]]


def is_valid_port(port, ship, u=None):
    r = ship["r"]
    l = ship["l"]
    d = ship["d"]
    p = ship["p"]
    b = port["b"]
    D = port["D"]
    if l > b:  # ship length is larger than berth length
        return False
    return True


def is_valid_free(port_id, ship, u):
    global _port_free
    r = ship["r"]
    l = ship["l"]
    d = ship["d"]
    p = ship["p"]
    for i in range(p):
        if _port_free[port_id][u + i] == False:
            return False
    return True


def is_valid(port, ship, u):
    if not is_valid_port(port, ship, u):
        return False
    r = ship["r"]
    l = ship["l"]
    d = ship["d"]
    p = ship["p"]
    b = port["b"]
    D = port["D"]
    for i in range(p):
        if calD(u + i, D) < d:  # ship draft is larger than water depth
            return False
    return True


def greedy_solver(ports: pd.DataFrame, ships: pd.DataFrame, T=1440, a=2):
    global _port_free
    ships = ships.copy()
    ships = ships.sort_values(by="r")
    max_T = 10000
    _port_free = np.ones((len(ports), max_T), dtype=bool)
    result_port = []
    result_time = []
    result_k = []
    result_id = []

    for _, ship in tqdm(ships.iterrows()):
        count = 0
        tmp_time = np.full(len(ports), max_T)
        rand_eps = np.random.randint(0, len(ports))
        for rawI, _ in ports.iterrows():
            i = (rawI + rand_eps) % len(ports)
            port = ports.loc[i]
            if not is_valid_port(port, ship, None):
                continue
            for u in range(ship["r"], max_T):
                if is_valid(port, ship, u) and is_valid_free(i, ship, u):
                    tmp_time[i] = u
                    count += 1
                    break
            if count >= 2:
                break
        argmin = np.argmin(tmp_time)
        result_port.append(argmin)
        result_time.append(tmp_time[argmin])
        result_k.append(-1)
        result_id.append(ship["id"])
        for j in range(tmp_time[argmin], tmp_time[argmin] + ship["p"]):
            _port_free[argmin][j] = False

    result = pd.DataFrame(
        {
            "id": result_id,
            "port": result_port,
            "k": result_k,
            "arrival_time": ships["r"].values,
            "start_time": result_time,
        }
    )
    result["end_time"] = result["start_time"].values + ships["p"].values
    result["delay_time"] = result["start_time"] - result["arrival_time"]
    for i in range(len(result)):
        count = 0
        for j in range(len(result)):
            if (
                result.loc[j, "start_time"] < result.loc[i, "start_time"]
                and result.loc[i, "port"] == result.loc[j, "port"]
            ):
                count += 1
        result.loc[i, "k"] = count
    result["l"] = ships["l"].values
    result["b"] = result["port"].apply(lambda x: ports.loc[x, "b"])
    tmp_result = result.copy()
    tmp_result.sort_values(by=["port", "start_time"], inplace=True)
    print(tmp_result)
    print(tmp_result["delay_time"].sum())
    result.sort_values(by="id", inplace=True)
    return result


def callback(model, where):
    if where == GRB.Callback.MIP:
        # 获取当前的最优解
        obj_val = model.cbGet(GRB.Callback.MIP_OBJBST)

        # 输出当前找到的最优解
        print("当前最优解:", obj_val)


def solver(
    ports: pd.DataFrame,
    ships: pd.DataFrame,
    T=1440,
    a=2,
    output="output.csv",
    greedy_result=None,
):
    """
    \min \sum_{i \in V}(t_i - r_i) \\
    \text{s.t.} \\
    \begin {align} 
    \sum_{j \in B, k \in O} x_{ijk} = 1, \forall i \in V \\
    \sum_{i \in V} x_{ijk} \leq 1, \forall j \in B, k \in O \\
    r_i - t_i \leq 0, \forall i \in V \\
    \sum_{i \in V} x_{ijk} - \sum_{i \in V} x_{ijk+1} \leq 0, \forall j \in B, k,k+1 \in O \\
    x_{ijk}x_{ij'k+1}(t_i + p_i - t_{i'}) \leq 0, \forall i, i' \in V, j \in B, k \in O \\
    x_{ijk}(l_i-b_j) \leq 0, \forall i \in V, j \in B, k \in O \\
    x_{ijk}(d_i - D_j^{t_i+u}) \leq 0, \forall i \in V, j \in B, k \in O, u \in P_i \\
    x_{ijk} \in \{0, 1\}, \forall i \in V, j \in B, k \in O \\
    t_i \geq 0, \forall i \in V
    """
    r = ships["r"].values
    l = ships["l"].values
    d = ships["d"].values
    p = ships["p"].values
    b = ports["b"].values
    D = ports["D"].values
    V = np.arange(len(ships))
    B = np.arange(len(ports))
    O = V.copy()
    step = 100
    working_time_max = p.max()
    model = gp.Model("SBSP")
    x = model.addVars(V, B, O, vtype=GRB.BINARY, name="x")
    t = model.addVars(V, vtype=GRB.CONTINUOUS, name="t")

    for x_i in x.values():
        x_i.Start = 0

    x_dict = {}
    t_dict = {}

    if isinstance(greedy_result, pd.DataFrame):
        for _, row in greedy_result.iterrows():
            i = row["id"] - 1
            j = row["port"]
            k = row["k"]
            x[i, j, k].Start = 1
            t[i].Start = row["start_time"]
            x_dict[(i, j, k)] = 1
            t_dict[i] = row["start_time"]

    model.setObjective(gp.quicksum(t[i] - r[i] for i in V), GRB.MINIMIZE)
    model.addConstrs(
        (gp.quicksum(x[i, j, k] for j in B for k in O) == 1 for i in V),
        name="one_ship_one_port",
    )
    model.addConstrs(
        (gp.quicksum(x[i, j, k] for i in V) <= 1 for j in B for k in O),
        name="one_port_one_ship",
    )
    model.addConstrs((t[i] >= r[i] for i in V), name="ship_arrive_before_ready")
    model.addConstrs(
        (
            gp.quicksum(x[i, j, k] for i in V) - gp.quicksum(x[i, j, k + 1] for i in V)
            >= 0
            for j in B
            for k in range(len(O) - 1)
        ),
        name="ship_not_overlap",
    )
    mulx = model.addVars(V, V, B, O, vtype=GRB.BINARY, name="mulx")
    for i, i_, j, k in mulx.keys():
        if i != i_ and k < len(O) - 1:
            x1 = x_dict.get((i, j, k), 0)
            x2 = x_dict.get((i_, j, k + 1), 0)
            mulx[i, i_, j, k].Start = x1 * x2
    model.addConstrs(
        (
            mulx[i, i_, j, k] == x[i, j, k] * x[i_, j, k + 1]
            for i in V
            for i_ in V
            for j in B
            for k in range(len(O) - 1)
            if i != i_
        ),
        name="mulx",
    )
    model.addConstrs(
        (
            mulx[i, i_, j, k] * (t[i] + p[i] - t[i_]) <= 0
            for i in V
            for i_ in V
            for j in B
            for k in O
            if i != i_
        ),
        name="ship_not_overlap",
    )
    model.addConstrs(
        (x[i, j, k] * (l[i] - b[j]) <= 0 for i in V for j in B for k in O),
        name="ship_not_exceed_port",
    )

    # phase = model.addVars(
    #     V,
    #     range(working_time_max),
    #     vtype=GRB.CONTINUOUS,
    #     name="phase",
    # )
    # phase_dict = np.empty((len(V), working_time_max), dtype=object)
    # for i in V:
    #     for u in range(0, p[i], step):
    #         temp = 2 * np.pi * (t_dict.get(i, 0) + u) / T
    #         phase[i, u].Start = temp
    #         phase_dict[i, u] = temp

    # model.addConstrs(
    #     (
    #         phase[i, u] == 2 * np.pi * (t[i] + u) / T
    #         for i in V
    #         for u in range(0, p[i], step)
    #     ),
    #     name="phase",
    # )

    # tide = model.addVars(V, range(working_time_max), vtype=GRB.CONTINUOUS, name="tide")
    # for i in V:
    #     for u in range(0, p[i], step):
    #         tide[i, u].Start = np.sin(phase_dict[i, u])

    # for i in V:
    #     for u in range(0, p[i], step):
    #         model.addGenConstrSin(
    #             phase[i, u],
    #             tide[i, u],
    #             name="tide",
    #         )

    # model.addConstrs(
    #     (
    #         x[i, j, k] * (d[i] - a * tide[i, u] - D[j]) <= 0
    #         for i in V
    #         for j in B
    #         for k in O
    #         for u in range(0, p[i], step)
    #         if D[j] - a <= d[i]
    #     ),
    #     name="ship_not_exceed_depth",
    # )

    # model.setParam("TimeLimit", 60)
    # model.setParam("MIPGap", 0.2)
    model.optimize()  # callback=callback
    # solution
    print(f"Objective: {model.objVal}")
    if model.status == GRB.OPTIMAL:
        result_port = []
        result_time = []
        result_k = []
        for i in V:
            for j in B:
                for k in O:
                    if x[i, j, k].x > 0:
                        # print(
                        #     "Ship %d (arrive at %d) at Port %d at Time %d"
                        #     % (i, r[i], j, t[i].x)
                        # )
                        result_port.append(j)
                        result_time.append(t[i].x)
                        result_k.append(k)
        result = pd.DataFrame(
            {
                "port": result_port,
                "k": result_k,
                "arrival_time": r,
                "start_time": result_time,
            }
        )
        result["end_time"] = result["start_time"].values + ships["p"].values
        result["delay_time"] = result["start_time"] - result["arrival_time"]
        tmp_result = result.copy()
        tmp_result.sort_values(by=["port", "start_time"], inplace=True)
        print(tmp_result)
        result.to_csv(output, index=False)
        return result
    else:
        return None


def check(ports: pd.DataFrame, ships: pd.DataFrame, result: pd.DataFrame, T=1440, a=2):
    port_ship = [[] for _ in range(len(ports))]
    ships = pd.merge(ships, result, left_index=True, right_index=True)
    for i, ship in ships.iterrows():
        port_ship[int(ship["port"])].append(ship)
    port_ship = [sorted(ps, key=lambda x: x["start_time"]) for ps in port_ship]
    for i, ps in enumerate(port_ship):
        for j, ship in enumerate(ps):
            if j == 0:
                continue
            if ship["start_time"] < ps[j - 1]["start_time"] + ps[j - 1]["p"]:
                print(f"Port {i} Ship {j} is not feasible")
                print(ps[j - 1])
                print(ship)
                return False


def plot(ports: pd.DataFrame, ships: pd.DataFrame, result: pd.DataFrame, T=1440, a=2):
    maxT = (result["start_time"].values + ships["p"].values).max().astype(int)
    t = np.linspace(0, maxT, maxT)
    d_list = np.array([d0 + 2 * np.sin(2 * np.pi * t / T) for d0 in ports["D"].values])
    rectangle_height = ports["b"].values
    rectangle_y0 = np.cumsum(rectangle_height) - rectangle_height
    rectangle_y1 = np.cumsum(rectangle_height)
    rectangle_mid = (rectangle_y0 + rectangle_y1) / 2
    d_map = np.vstack(
        [
            np.repeat(d.reshape(1, -1), h, axis=0)
            for d, h in zip(d_list, rectangle_height)
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 20))
    fig.colorbar(
        ax.imshow(
            d_map,
        )
    )
    ax.set_yticks(
        rectangle_mid,
        labels=["{}({})".format(i + 1, d) for i, d in enumerate(rectangle_height)],
    )
    for h in rectangle_y0:
        ax.axhline(h, color="k")
    ax.set_aspect("auto")
    # ax.fill_between
    st_time = result["start_time"].values
    ed_time = st_time + ships["p"].values
    small_l = ships["l"].values * 0.5
    ship_y0 = np.array(
        [rectangle_mid[p] - l / 2 for p, l in zip(result["port"].values, small_l)]
    )
    ship_y1 = ship_y0 + small_l
    for x0, x1, y0, y1 in zip(st_time, ed_time, ship_y0, ship_y1):
        ax.fill_between([x0, x1], [y0, y0], [y1, y1], color="gray")
        ax.plot([x0, x0], [y0, y1], color="k", linestyle="--")
        ax.plot([x1, x1], [y0, y1], color="k", linestyle="--")

    fig.savefig("tide.png")


def main():
    # b: 泊位长度; D: 初始水位
    ports = pd.read_csv(PORT_DATA_FILE, header=None, names=["id", "b", "D"])
    # r: 到达时间; p: 工作时间; l: 船长; d: 吃水深度
    ships_list = {
        k: pd.read_csv(file, header=None, names=["id", "r", "p", "l", "d"])
        for k, file in SHIP_DATA_FILES.items()
    }
    if arg.ship_dataset == "all":
        for k, ships in ships_list.items():
            print(f"Dataset: {k}")
            solver(ports, ships, T=1440, a=2, output=f"output-{k}.csv")
    elif arg.ship_dataset == "tiny":
        ships = pd.read_csv(
            SHIP_TINY_DATA_FILE, header=None, names=["id", "r", "p", "l", "d"]
        )
        greedy_result = greedy_solver(ports, ships, T=1440, a=2)
        solver(
            ports,
            ships,
            T=1440,
            a=2,
            output=f"output-tiny.csv",
            greedy_result=greedy_result,
        )
    else:
        ships = ships_list[arg.ship_dataset]
        # check(ports, ships, pd.read_csv(f"output-{arg.ship_dataset}.csv"))
        print(f"Dataset: {arg.ship_dataset}")
        # solver(ports, ships, T=1440, a=2, output=f"output-{arg.ship_dataset}.csv")
        greedy_solver(ports, ships, T=1440, a=2)


main()
