#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "argparse.hpp"
using namespace std;
const int a = 2;
const int T = 1440;
const float pi = 3.1415926;
int port_num = 20;
int ship_num;
argparse::ArgumentParser parser("ga", "Genetic Algorithm");
// output
// port,arrival_time,start_time,id,delay_time
struct Chromosome {
    vector<int> priority_gene;
    vector<int> port_gene;
};
struct Port {
    int id;
    int length;
    int depth;
};
struct Ship {
    int arrival_time;
    int work_time;
    int length;
    int depth;
};

/*
 * @brief 下艘船尽早安排的时间，D_0 + a * sin( 2 * pi * x / T ) = t { x>=u }
 * @param current_time 当前时间
 * @param D 该港口的初始水深
 * @param d 该船的吃水
 * @param work_time 该船的作业时间
 */
int calc_start_time(int current_time, int D, int d, int work_time) {
    int delta = d - D;  // -2, -1, 0, 1
    if (delta <= -2) {
        return current_time;
    }
    static const int step = pi / 6 * T;
    static const int st[] = {-step, 0, step}, ed[] = {7 * step, 6 * step, 5 * step};
    int idx = delta + 1;
    int u = (current_time - st[idx]) % T + st[idx];  // u = [st, st + T)
    int n = (current_time - st[idx]) / T;
    if (ed[idx] - u < work_time) {
        return st[idx] + (n + 1) * T;
    } else {
        return current_time;
    }
}

int calc_fitness(const Chromosome& c) {
    int current_time[port_num] = {0};
    return 0;
}
int main(int argc, char* argv[]) {
    parser.add_argument("-d", "--dataset")
        .help("ships<dataset>.txt")
        .default_value(string("20"))
        .required();
    parser.add_argument("-o", "--output")
        .help("output directory")
        .default_value(string("../result/ga"))
        .required();
    try {
        parser.parse_args(argc, argv);
    } catch (const runtime_error& err) {
        cout << err.what() << endl;
        cout << parser;
        exit(0);
    }
    string dataset = parser.get<string>("--dataset");
    string dataset_path = "../data/ships" + dataset + ".txt";
    string output_dir = parser.get<string>("--output");
    string output_path = output_dir + "/output-" + dataset + ".csv";
    // cout << "dataset: " << dataset_path << endl;
    // cout << "output: " << output_path << endl;
    cout << calc_start_time(0, 0, 0, 0) << endl;
}