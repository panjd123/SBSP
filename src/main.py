import pandas as pd
import os.path as osp
import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--ship_dataset", "-d", type=str, default="20", required=False)
parser.add_argument("--time_limit", "-t", type=int, default=600, required=False)
parser.add_argument("--result_dir", "-o", type=str, default="result", required=False)
parser.add_argument("--no_solve", "-n", action="store_true", required=False)
parser.add_argument("--continue_solve", "-c", action="store_true", required=False)
parser.add_argument("--greedy_total", "-g", type=int, default=100, required=False)
parser.add_argument("--simple_draft", "-f", action="store_true", required=False)
parser.add_argument("--approx_draft", "-a", action="store_true", required=False)
parser.add_argument("--tstep", "-s", type=int, default=1, required=False)
parser.add_argument("--time_threshold", "-r", type=int, default=9999, required=False)
parser.add_argument("--length_threshold", "-l", type=int, default=0, required=False)
args = parser.parse_args()

DATA_DIR = "../data/"
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
    return D_0 + a * np.sin(2 * np.pi * t / T)


# [x0, x1)
def get_piecewise_points(to_T=1440, T_num=10, breakpoint_only=False):
    eps = 1e-5
    x0 = [0, np.pi / 6, 5 * np.pi / 6, np.pi, 7 * np.pi / 6, 11 * np.pi / 6]
    x = np.repeat(x0, 2)
    x[np.arange(0, len(x), 2)] -= eps
    y0 = [0, 1, 0, -1, -2, -1]
    y = np.repeat(y0, 2)
    x = np.tile(x, T_num).reshape(T_num, -1)
    x_base = np.arange(0, T_num) * 2 * np.pi
    x = x + x_base.reshape(T_num, 1)
    x = x.flatten()
    y = np.tile(y, T_num)
    x = x[1:]
    y = y[:-1]
    if type(to_T) == int:
        x = x * to_T / (2 * np.pi)
    if breakpoint_only:
        return x[::2], y[::2]
    else:
        return x, y


def piecewise_linear(x, xp, yp):
    slopes = np.diff(yp) / np.diff(xp)
    if x <= xp[0]:
        return yp[0] + slopes[0] * (x - xp[0])
    if x >= xp[-1]:
        return yp[-1] + slopes[-1] * (x - xp[-1])
    i = np.searchsorted(xp, x) - 1
    return yp[i] + slopes[i] * (x - xp[i])


def get_tide_min():
    _, y = get_piecewise_points(T_num=2, breakpoint_only=True)
    tide_min = np.full((6, 6, 2), 100)
    for o in [0, 1]:
        for i in range(6):
            for j in range(6):
                tb = i
                te = j
                if o == 1:
                    te += 6
                if tb > te:
                    continue
                t = np.min(y[tb : te + 1])
                tide_min[i, j, o] = t
    return tide_min


def greedy_solver(ports: pd.DataFrame, ships: pd.DataFrame, T=1440, a=2):
    max_T = 10000
    fail_count = 0
    result_list = []
    total = args.greedy_total
    tq = tqdm(total=total)
    while True:
        try:
            result = greedy_solver_it(ports, ships, T, a, max_T=max_T)
            result_list.append(result)
            tq.update(1)
            if len(result_list) > total:
                r_min = min(result_list, key=lambda x: x["delay_time"].mean())
                print(r_min.sort_values(by=["port", "start_time"]))
                print(r_min["delay_time"].mean())
                return r_min
        except Exception as e:
            fail_count += 1
            if fail_count > 10 and len(result_list) < 1:
                if max_T > 160000:
                    raise Exception("fail to find a solution")
                else:
                    max_T *= 2
                    fail_count = 0


# D_0 + a * sin( 2 * pi * x / T ) = t { x>=u }
def first_meet_point(u, t, D_0, a=2, T=1440):
    t = (t - D_0) / a
    t = 2 * np.pi * t / T
    x = np.arcsin(t)
    # x + 2*pi * (k-1) < u <= x + 2*pi * k
    k = np.ceil((u - x) / (2 * np.pi))
    x = x + 2 * np.pi * k
    return x * T / (2 * np.pi)


def greedy_solver_it(
    ports: pd.DataFrame, ships: pd.DataFrame, T=1440, a=2, max_T=10000
):
    ships = ships.copy()
    ships = ships.sort_values(by="r")
    _port_free_list = [[[0, max_T]] for _ in range(len(ports))]
    result_port = []
    result_time = []
    result_id = []

    for _, ship in ships.iterrows():
        tmp_ans = []
        rand_eps = np.random.randint(0, len(ports))
        for rawI, _ in ports.iterrows():
            i = (rawI + rand_eps) % len(ports)
            port = ports.loc[i]
            if port["b"] < ship["l"] or port["D"] + a < ship["d"]:
                continue
            for node_index in range(len(_port_free_list[i])):
                list_node = _port_free_list[i][node_index]
                bg, ed = list_node
                true_bg = max(bg, ship["r"])
                if true_bg + ship["p"] > ed:
                    continue
                if port["D"] - a < ship["d"]:
                    true_bg = first_meet_point(true_bg, ship["d"], port["D"], a, T)
                    true_ed = true_bg + ship["p"]
                    if true_ed > ed or calD(true_ed, port["D"], a, T) < ship["d"]:
                        continue
                    first_min_point = first_meet_point(true_bg, -1, port["D"], a, T)
                    if first_min_point < true_bg:
                        continue
                tmp_ans.append([i, true_bg, node_index])
                break
            if len(tmp_ans) > 10:
                break
        if len(tmp_ans) < 1:
            raise Exception("fail to find a solution")
        min_time = min(tmp_ans, key=lambda x: x[1])
        port_index = min_time[0]
        u = min_time[1]
        node_index = min_time[2]
        node = _port_free_list[port_index][node_index]
        bg, ed = node
        _port_free_list[port_index].pop(node_index)

        if u > bg:
            _port_free_list[port_index].insert(node_index, [bg, u])
            node_index += 1
        if ed > u + ship["p"]:
            _port_free_list[port_index].insert(node_index, [u + ship["p"], ed])
            node_index += 1
        result_port.append(port_index)
        result_time.append(u)
        result_id.append(ship["id"])

    result = pd.DataFrame(
        {
            "id": result_id,
            "port": result_port,
            "arrival_time": ships["r"].values,
            "start_time": result_time,
        }
    )
    result["end_time"] = result["start_time"].values + ships["p"].values
    result["delay_time"] = result["start_time"] - result["arrival_time"]
    result["l"] = ships["l"].values
    result["b"] = result["port"].apply(lambda x: ports.loc[x, "b"])
    # tmp_result = result.copy()
    # tmp_result.sort_values(by=["port", "start_time"], inplace=True)
    # print(tmp_result)
    # print(tmp_result["delay_time"].mean())
    result.sort_values(by="id", inplace=True)
    check(ports, ships.sort_values("id").reset_index(), result.reset_index())
    return result


def extract_result(x, t, ships):
    result_port = []
    result_time = []
    for i, j in x.keys():
        if np.isclose(x[i, j], 1):
            result_port.append(int(j))
            result_time.append(round(t[i]))
    result = pd.DataFrame(
        {
            "port": result_port,
            "arrival_time": ships["r"].values,
            "start_time": result_time,
        }
    )
    result["id"] = ships["id"].values
    result["delay_time"] = result["start_time"] - result["arrival_time"]
    tmp_result = result.copy()
    tmp_result.sort_values(by=["port", "start_time"], inplace=True)
    print(tmp_result)
    print("Mean delay:{:.2f}".format(tmp_result["delay_time"].mean()))
    return result


_tmp_obj_result = None


def callback(model, where):
    global _tmp_obj_result
    if where == GRB.Callback.MIPSOL:
        _x = model.cbGetSolution(model._x)
        _t = model.cbGetSolution(model._t)
        if not _tmp_obj_result:
            print("Gurobi: First valid solution found.")
        _tmp_obj_result = (_x, _t)


def simplex_solver(
    ports: pd.DataFrame,
    ships: pd.DataFrame,
    T=1440,
    a=2,
    output="output.csv",
    greedy_result=None,
):
    r = ships["r"].values
    l = ships["l"].values
    d = ships["d"].values
    p = ships["p"].values
    b = ports["b"].values
    D = ports["D"].values
    V = np.arange(len(ships))
    B = np.arange(len(ports))
    working_time_max = p.max()
    M = 1e5
    model = gp.Model("SBSP")
    x = model.addVars(V, B, vtype=GRB.BINARY, name="x")
    t = model.addVars(V, lb=0, ub=1440 * 25, vtype=GRB.CONTINUOUS, name="t")
    y = model.addVars(V, V, vtype=GRB.BINARY, name="y")

    for x_i in x.values():
        x_i.Start = 0

    x_dict = {}
    t_dict = {}

    ship_port = {}

    if isinstance(greedy_result, pd.DataFrame):
        for _, row in greedy_result.iterrows():
            i = row["id"] - 1
            j = row["port"]
            x[i, j].Start = 1
            t[i].Start = row["start_time"]
            x_dict[(i, j)] = 1
            t_dict[i] = row["start_time"]
            # print("ship {} start at port {} at time {}".format(i, j, row["start_time"]))
            ship_port[i] = j

    model.setObjective(gp.quicksum(t[i] - r[i] for i in V), GRB.MINIMIZE)
    model.addConstrs(
        (gp.quicksum(x[i, j] for j in B) == 1 for i in V),
        name="one_ship_one_port",
    )
    model.addConstrs((t[i] >= r[i] for i in V), name="ship_arrive_before_ready")

    mulx = model.addVars(V, V, B, vtype=GRB.BINARY, name="mulx")
    for i, i_, j in mulx.keys():
        if i != i_:
            x1 = x_dict.get((i, j), 0)
            x2 = x_dict.get((i_, j), 0)
            mulx[i, i_, j].Start = x1 * x2

    model.addConstrs(
        (
            mulx[i, i_, j] == x[i, j] * x[i_, j]
            for i in V
            for i_ in range(i + 1, len(V))
            for j in B
            if i != i_
        ),
        name="mulx",
    )

    model.addConstrs(
        (
            mulx[i, i_, j] * (t[i] + p[i] - t[i_]) <= M * (1 - y[i, i_])
            for i in V
            for i_ in range(i + 1, len(V))
            for j in B
        ),
        name="ship_not_overlap",
    )

    model.addConstrs(
        (
            mulx[i, i_, j] * (t[i_] + p[i_] - t[i]) <= M * y[i, i_]
            for i in V
            for i_ in range(i + 1, len(V))
            for j in B
        ),
        name="ship_not_overlap",
    )

    model.addConstrs(
        (x[i, j] * (l[i] - b[j]) <= 0 for i in V for j in B),
        name="ship_not_exceed_port",
    )

    #### draft constraint ####

    # simple draft constraint
    if args.simple_draft:
        model.addConstrs(
            (x[i, j] == 0 for i in V for j in B if D[j] - a < d[i]),
            name="ship_not_exceed_depth",
        )
    elif args.approx_draft:
        L = args.length_threshold

        phase = model.addVars(
            V,
            range(working_time_max),
            lb=0,
            ub=50 * np.pi,
            vtype=GRB.CONTINUOUS,
            name="phase",
        )

        phase_dict = np.empty((len(V), working_time_max), dtype=object)
        for i in V:
            for u in range(0, p[i]):
                temp = 2 * np.pi * (t_dict.get(i, 0) + u) / T
                phase[i, u].Start = temp
                phase_dict[i, u] = temp

        model.addConstrs(
            (
                phase[i, u] == 2 * np.pi * (t[i] + u) / T
                for i in V
                for u in range(0, p[i])
                if l[i] >= L
            ),
            name="phase",
        )

        tide = model.addVars(
            V, range(working_time_max), lb=-2, ub=2, vtype=GRB.CONTINUOUS, name="tide"
        )

        xp, yp = get_piecewise_points(None, 15)
        xp, yp = np.arange(0, 20 * np.pi, 0.5 * np.pi), np.tile(
            np.array([0, 1, 0, -np.pi / 2]), 10
        )

        for i in V:
            for u in range(0, p[i]):
                tide[i, u].Start = piecewise_linear(phase_dict[i, u], xp, yp)

        for i in V:
            if l[i] >= L:
                for u in range(0, p[i]):
                    model.addGenConstrPWL(
                        phase[i, u],
                        tide[i, u],
                        xp,
                        yp,
                    )

        # original "ship_not_exceed_depth"
        model.addConstrs(
            (
                x[i, j] * (d[i] - (D[j] + a * tide[i, u])) <= 0  # tide[i, u]
                for i in V
                for j in B
                for u in range(0, p[i])
                if D[j] - a <= d[i]
            ),
            name="ship_not_exceed_depth",
        )
        # approximate "ship_not_exceed_depth"
        # # step = 0.5 * np.pi
        # # tStep = max(1, int(step / (2 * np.pi) * T))
        # tStep = args.tstep
        # step = tStep * 2 * np.pi / T if tStep > 1 else 0
        # threshold = args.time_threshold
        # assert step <= np.pi
        # model.addConstrs(
        #     (
        #         x[i, j] * (d[i] - (D[j] + a * (tide[i, u] - 2 * np.sin(step / 2))))
        #         <= 0  # tide[i, u] <-> -1
        #         for i in V
        #         for j in B
        #         for u in range(0, p[i], tStep)
        #         if D[j] - a <= d[i] and p[i] <= threshold and l[i] >= L
        #     ),
        #     name="ship_not_exceed_depth",
        # )

        # model.addConstrs(
        #     (
        #         x[i, j] == 0
        #         for i in V
        #         for j in B
        #         if D[j] - a < d[i] and (p[i] > threshold or l[i] < L)
        #     ),
        #     name="ship_not_exceed_depth",
        # )
    else:
        qbg = model.addVars(V, vtype=GRB.INTEGER, name="qbg")
        qed = model.addVars(V, vtype=GRB.INTEGER, name="qed")
        rbg = model.addVars(V, lb=0, ub=T, name="rbg")
        red = model.addVars(V, lb=0, ub=T, name="red")
        model.addConstrs(
            (t[i] == qbg[i] * T + rbg[i] for i in V),
            name="tbg_in_cycle",
        )
        model.addConstrs(
            (t[i] + p[i] == qed[i] * T + red[i] for i in V),
            name="ted_in_cycle",
        )

        for i in V:
            qbg[i].Start = t_dict.get(i) // T
            rbg[i].Start = t_dict.get(i) % T
            qed[i].Start = (t_dict.get(i) + p[i]) // T
            red[i].Start = (t_dict.get(i) + p[i]) % T

        breakpoint, _ = get_piecewise_points(breakpoint_only=True)
        # SOS2
        wbg = model.addVars(V, range(7), vtype=GRB.CONTINUOUS, name="wbg")
        zbg = model.addVars(V, range(6), vtype=GRB.BINARY, name="zbg")
        model.addConstrs(
            (
                rbg[i] == gp.quicksum(wbg[i, j] * breakpoint[j] for j in range(7))
                for i in V
            ),
            name="rbg",
        )
        model.addConstrs(
            (gp.quicksum(wbg[i, j] for j in range(7)) == 1 for i in V),
            name="wbg_one",
        )
        model.addConstrs(
            (gp.quicksum(zbg[i, j] for j in range(6)) == 1 for i in V),
            name="zbg_one",
        )
        model.addConstrs(
            (
                wbg[i, j]
                <= (zbg[i, j - 1] if j >= 1 else 0) + (zbg[i, j] if j < 6 else 0)
                for i in V
                for j in range(7)
            )
        )
        wed = model.addVars(V, range(7), vtype=GRB.CONTINUOUS, name="wed")
        zed = model.addVars(V, range(6), vtype=GRB.BINARY, name="zed")
        model.addConstrs(
            (
                red[i] == gp.quicksum(wed[i, j] * breakpoint[j] for j in range(7))
                for i in V
            ),
            name="red",
        )
        model.addConstrs(
            (gp.quicksum(wed[i, j] for j in range(7)) == 1 for i in V),
            name="wed_one",
        )
        model.addConstrs(
            (gp.quicksum(zed[i, j] for j in range(6)) == 1 for i in V),
            name="zed_one",
        )
        model.addConstrs(
            (
                wed[i, j]
                <= (zed[i, j - 1] if j >= 1 else 0) + (zed[i, j] if j < 6 else 0)
                for i in V
                for j in range(7)
            )
        )
        o = model.addVars(
            V,
            vtype=GRB.BINARY,
            name="o",
        )
        model.addConstrs(
            (o[i] == qed[i] - qbg[i] for i in V),
            name="o",
        )

        tide_min = model.addVars(V, lb=-2, ub=1)
        tide_min0 = model.addVars(V, lb=-2, ub=1)
        tide_min1 = model.addVars(V, lb=-2, ub=1)
        f = get_tide_min()
        model.addConstrs(
            (
                tide_min0[i]
                == quicksum(
                    zbg[i, j] * zed[i, k] * f[j, k, 0]
                    for j in range(6)
                    for k in range(6)
                    if j <= k
                )
                for i in V
            )
        )
        model.addConstrs(
            (
                tide_min1[i]
                == quicksum(
                    zbg[i, j] * zed[i, k] * f[j, k, 1]
                    for j in range(6)
                    for k in range(6)
                    if j >= k
                )
                for i in V
            )
        )
        model.addConstrs(
            (tide_min[i] == tide_min0[i] * (1 - o[i]) + tide_min1[i] * o[i] for i in V)
        )

        # Start
        for i in V:
            t_bg = t_dict.get(i, 0) % T
            t_ed = (t_dict.get(i, 0) + p[i]) % T
            k0 = 0
            k1 = 0
            for j in range(6):
                if breakpoint[j] <= t_bg < breakpoint[j + 1]:
                    zbg[i, j].Start = 1
                    k0 = j
                    p1 = (breakpoint[j + 1] - t_bg) / (
                        breakpoint[j + 1] - breakpoint[j]
                    )
                    p2 = 1 - p1
                    wbg[i, j].Start = p1
                    wbg[i, j + 1].Start = p2
                    for k in range(7):
                        if k == j or k == j + 1:
                            continue
                        else:
                            wbg[i, k].Start = 0
                else:
                    zbg[i, j].Start = 0

                if breakpoint[j] <= t_ed < breakpoint[j + 1]:
                    k1 = j
                    zed[i, j].Start = 1
                    p1 = (breakpoint[j + 1] - t_ed) / (
                        breakpoint[j + 1] - breakpoint[j]
                    )
                    p2 = 1 - p1
                    wed[i, j].Start = p1
                    wed[i, j + 1].Start = p2
                    for k in range(7):
                        if k == j or k == j + 1:
                            continue
                        else:
                            wed[i, k].Start = 0
                else:
                    zed[i, j].Start = 0

            o_s = (t_dict.get(i, 0) + p[i]) // T - t_dict.get(i, 0) // T
            o[i].Start = o_s
            tide_min0[i].Start = f[k0, k1, 0]
            tide_min1[i].Start = f[k0, k1, 1]
            tide_min[i].Start = f[k0, k1, 0] * (1 - o_s) + f[k0, k1, 1] * o_s
            tide_s = f[k0, k1, 0] * (1 - o_s) + f[k0, k1, 1] * o_s
            assert d[i] - (D[ship_port[i]] + tide_s) <= 0

        threshold = args.time_threshold
        L = args.length_threshold

        model.addConstrs(
            (
                x[i, j] * (d[i] - (D[j] + tide_min[i])) <= 0
                for i in V
                for j in B
                if D[j] - a <= d[i] and p[i] <= threshold and l[i] >= L
            ),
            name="ship_not_exceed_depth",
        )

        model.addConstrs(
            (
                x[i, j] == 0
                for i in V
                for j in B
                if D[j] - a < d[i] and (p[i] > threshold or l[i] < L)
            ),
            name="ship_not_exceed_depth",
        )

    #### draft constraint end ####

    model.setParam("TimeLimit", args.time_limit)
    model.update()
    model._x = x
    model._t = t
    model._y = y
    model.optimize(callback=callback)
    print(f"Objective: {model.objVal}")

    if model.status == GRB.OPTIMAL:
        _x = {id: x[id].x for id in x.keys()}
        _t = {id: t[id].x for id in t.keys()}
        result = extract_result(_x, _t, ships)
    else:
        if not _tmp_obj_result:
            print("No feasible solution found.")
            return None
        result = extract_result(_tmp_obj_result[0], _tmp_obj_result[1], ships)

    result.to_csv(output, index=False)
    print(result["delay_time"].mean())
    return result


def check(ports: pd.DataFrame, ships: pd.DataFrame, result: pd.DataFrame, T=1440, a=2):
    port_ship = [[] for _ in range(len(ports))]
    ships = pd.merge(ships, result, left_index=True, right_index=True)
    ships["end_time"] = ships["start_time"] + ships["p"]
    for i, ship in ships.iterrows():
        port_ship[int(ship["port"])].append(ship)
    port_ship = [sorted(ps, key=lambda x: x["start_time"]) for ps in port_ship]
    for i, ps in enumerate(port_ship):
        for j, ship in enumerate(ps):
            if j >= 1 and ship["start_time"] < ps[j - 1]["end_time"]:
                print(f"overlap")
                print(ps[j - 1])
                print(ship)
                return False

            for u in range(0, ship["p"]):
                t = ship["start_time"] + u
                port_id = int(ship["port"])
                D = ports.loc[port_id, "D"] + a * np.sin(2 * np.pi * t / T)
                if D + 0.1 < ship["d"]:
                    print("draft {} > {}".format(ship["d"], D))
                    print(ports.iloc[port_id])
                    print(ship)
                    return False
    return True


def plot(
    ports: pd.DataFrame,
    ships: pd.DataFrame,
    result: pd.DataFrame,
    T=1440,
    a=2,
    output="plot.png",
):
    if not isinstance(result, pd.DataFrame):
        return
    maxT = (result["start_time"].values + ships["p"].values).max().astype(int)
    t = np.linspace(0, maxT, maxT)
    d_list = np.array([d0 + a * np.sin(2 * np.pi * t / T) for d0 in ports["D"].values])
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
    fig, ax = plt.subplots(figsize=(10, 14))
    fig.colorbar(
        ax.imshow(
            d_map,
            cmap="Blues",
        )
    )
    ax.set_yticks(
        rectangle_mid,
        labels=["{}({})".format(i + 1, d) for i, d in enumerate(rectangle_height)],
    )
    for h in rectangle_y0:
        ax.axhline(h, color="k")
    ax.set_aspect("auto")
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
    ax.set_xlim(0, maxT)
    fig.tight_layout()
    fig.savefig(output)


def solver(args):
    global _tmp_obj_result
    ports = args.ports
    ships = args.ships
    result_output_file = osp.join(args.result_dir, f"output-{args.ship_dataset}.csv")
    plot_output_file = osp.join(args.result_dir, f"plot-{args.ship_dataset}.png")
    print(f"Dataset: {args.ship_dataset}")
    if args.no_solve:
        result = pd.read_csv(result_output_file)
    else:
        if args.continue_solve and osp.exists(result_output_file):
            greedy_result = pd.read_csv(result_output_file)
        else:
            greedy_result = greedy_solver(ports, ships, T=1440, a=2)
        result = simplex_solver(
            ports,
            ships,
            output=result_output_file,
            greedy_result=greedy_result,
        )
    check(ports, ships, result)
    plot(
        ports,
        ships,
        result,
        output=plot_output_file,
    )
    _tmp_obj_result = None


def main():
    # b: 泊位长度; D: 初始水位
    ports = pd.read_csv(PORT_DATA_FILE, header=None, names=["id", "b", "D"])
    # r: 到达时间; p: 工作时间; l: 船长; d: 吃水深度
    ships_list = {
        k: pd.read_csv(file, header=None, names=["id", "r", "p", "l", "d"])
        for k, file in SHIP_DATA_FILES.items()
    }
    args.ports = ports

    if args.ship_dataset == "all":
        for k, ships in ships_list.items():
            args.ship_dataset = k
            args.ships = ships
            solver(args)
    elif args.ship_dataset == "tiny":
        ships = pd.read_csv(
            SHIP_TINY_DATA_FILE, header=None, names=["id", "r", "p", "l", "d"]
        )
        args.ships = ships
        solver(args)
    else:
        ships = ships_list[args.ship_dataset]
        args.ships = ships
        solver(args)


main()
