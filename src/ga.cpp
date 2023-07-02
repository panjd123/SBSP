#include <algorithm>
#include <cassert>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include "argparse.hpp"
using namespace std;
constexpr const int a = 2;
constexpr const int T = 1440;
constexpr const float pi = 3.1415926;
const int port_num = 20;
int ship_num;
int population_size = 10000;
int max_generation = 1000;
int mutation_rate = 0.3;
int mutation_num;
double truncation_rate = 0.1;
int truncation_size;
int tournament_size = 30;
int cross_point_num = -1;
bool break_tag;
int thread_num = 3;
int round_num = 1;
enum class SelectionMethod { Roulette,
                             Tournament,
                             Truncation
};
SelectionMethod selection_method = SelectionMethod::Tournament;
vector<vector<int>> available_ports_list;
argparse::ArgumentParser parser("ga", "Genetic Algorithm");

double quick_power(double x, int p) {
    double res = 1;
    while (p) {
        if (p & 1) {
            res *= x;
        }
        x *= x;
        p >>= 1;
    }
    return res;
}

struct Chromosome;
int calc_fitness(const Chromosome& c);
struct Chromosome {
    vector<int> priority_gene;
    vector<int> port_gene;
    int fitness;
    Chromosome() {
        priority_gene.resize(ship_num);
        port_gene.resize(ship_num);
    }
    void update_fitness() {
        fitness = calc_fitness(*this);
    }
    bool operator<(const Chromosome& rhs) const {
        return fitness < rhs.fitness;
    }
    bool operator>(const Chromosome& rhs) const {
        return fitness > rhs.fitness;
    }
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
vector<Port> ports;
vector<Ship> ships;
/*
 * @brief 下艘船尽早安排的时间，D_0 + a * sin( 2 * pi * x / T ) = t { x>=u }
 * @param current_time 当前时间
 * @param D 该港口的初始水深
 * @param d 该船的吃水
 * @param work_time 该船的作业时间
 */
int calc_start_time(int current_time, const Port& port, const Ship& ship) {
    int delta = ship.depth - port.depth;
    if (delta <= -2) {
        return max(current_time, ship.arrival_time);
    }
    static constexpr const int step = T / 12;
    static constexpr const int st[] = {-step, 0, step}, ed[] = {7 * step, 6 * step, 5 * step};
    int idx = delta + 1;
    current_time = max(current_time, ship.arrival_time);
    int u = (current_time - st[idx]) % T + st[idx];  // u = [st, st + T)
    int n = (current_time - st[idx]) / T;
    if (u < st[idx]) {
        u += T;
        n--;
    }
    int next_time = 1e9;
    if (ship.work_time <= ed[idx] - st[idx]) {  // no jmp
        int tmax0 = ed[idx] - ship.work_time;
        int tmin1 = st[idx] + T;
        if (u < tmax0) {
            next_time = min(next_time, u + n * T);
        } else {
            next_time = min(next_time, tmin1 + n * T);
        }
    }
    if (ship.work_time >= st[idx] + T - ed[idx]) {     // jmp
        int tmin0 = st[idx] + T - ship.work_time;      // 本周期最早开始时间
        int tmax0 = ed[idx];                           // 本周期最晚开始时间
        int tmin1 = st[idx] + 2 * T - ship.work_time;  // 下周期最早开始时间
        if (u < tmax0) {
            next_time = min(next_time, max(tmin0, u) + n * T);
        } else {
            next_time = min(next_time, tmin1 + n * T);
        }
    }
    return next_time;
}

void output(const Chromosome& c, string filename) {
    FILE* fp = fopen(filename.c_str(), "w");
    cerr << "output to " << filename << endl;
    fprintf(fp, "port,arrival_time,start_time,id,delay_time\n");
    int current_time[port_num] = {0};
    int start_time[ship_num] = {0};
    int d[ship_num];
    for (int i = 0; i < ship_num; i++) {
        d[i] = i;
    }
    sort(d, d + ship_num, [&](int a, int b) {
        return c.priority_gene[a] < c.priority_gene[b];
    });
    for (int i = 0; i < ship_num; i++) {
        int ship_id = d[i];
        int port_id = c.port_gene[ship_id];
        start_time[ship_id] = calc_start_time(current_time[port_id],
                                              ports[port_id], ships[ship_id]);
        current_time[port_id] = start_time[ship_id] + ships[ship_id].work_time;
    }
    for (int i = 0; i < ship_num; i++) {
        fprintf(fp, "%d,%d,%d,%d,%d\n", c.port_gene[i], ships[i].arrival_time, start_time[i], i + 1,
                start_time[i] - ships[i].arrival_time);
    }
}

int calc_fitness(const Chromosome& c) {
    int current_time[port_num] = {0};
    int delay_time_sum = 0;
    int d[ship_num];
    for (int i = 0; i < ship_num; i++) {
        d[i] = i;
    }
    sort(d, d + ship_num, [&](int a, int b) {
        return c.priority_gene[a] < c.priority_gene[b];
    });
    for (int i = 0; i < ship_num; i++) {
        int ship_id = d[i];
        int port_id = c.port_gene[ship_id];
        int start_time = calc_start_time(current_time[port_id],
                                         ports[port_id], ships[ship_id]);
        current_time[port_id] = start_time + ships[ship_id].work_time;
        delay_time_sum += start_time - ships[ship_id].arrival_time;
    }
    return delay_time_sum;
}

vector<int> available_ports(const Ship& ship) {
    static constexpr const int step = T / 12;
    static constexpr const int st[] = {-step, 0, step}, ed[] = {7 * step, 6 * step, 5 * step};
    vector<int> ret;
    for (int i = 0; i < port_num; i++) {
        if (ports[i].length < ship.length)
            continue;
        int delta = ship.depth - ports[i].depth;
        if (delta >= 2) {
            continue;
        }
        if (delta <= -2) {
            ret.push_back(i);
            continue;
        }
        // no jmp
        int idx = delta + 1;
        int time = ed[idx] - st[idx];
        if (time >= ship.work_time) {
            ret.push_back(i);
            continue;
        }
        // jmp
        if (st[idx] + T - ed[idx] <= ship.work_time) {
            ret.push_back(i);
            continue;
        }
    }
    return ret;
}

Chromosome crossover(const Chromosome& p1, const Chromosome& p2, unsigned int* seed) {
    Chromosome child;
    if (cross_point_num > 0) {
        int r[cross_point_num + 2] = {0};
        for (int i = 1; i <= cross_point_num; i++) {
            r[i] = rand_r(seed) % ship_num;
        }
        r[cross_point_num + 1] = ship_num;
        sort(r + 1, r + cross_point_num + 1);
        for (int i = 0; i <= cross_point_num; i++) {
            for (int j = r[i]; j <= r[i + 1]; j++) {
                if (i & 1) {
                    child.port_gene[j] = p1.port_gene[j];
                } else {
                    child.port_gene[j] = p2.port_gene[j];
                }
            }
        }
    } else {
        for (int i = 0; i < ship_num; i++) {
            if (rand_r(seed) & 1) {
                child.port_gene[i] = p1.port_gene[i];
            } else {
                child.port_gene[i] = p2.port_gene[i];
            }
        }
    }
    int fromParent1[ship_num];
    for (int i = 0; i < ship_num; i++) {
        fromParent1[i] = 0;
    }
    int r1 = rand_r(seed) % ship_num;
    int r2 = rand_r(seed) % ship_num;
    if (r1 > r2)
        swap(r1, r2);
    for (int i = r1; i <= r2; i++) {
        child.priority_gene[i] = p1.priority_gene[i];
        fromParent1[p1.priority_gene[i]] = 1;
    }
    for (int i = 0, t = 0; i < ship_num; i++) {
        if (t == r1)
            t = r2 + 1;
        if (fromParent1[p2.priority_gene[i]])
            continue;
        child.priority_gene[t] = p2.priority_gene[i];
        t++;
    }
    return child;
}

void mutation(Chromosome& c, unsigned int* seed) {
    int r1 = rand_r(seed) % ship_num;
    int r2 = rand_r(seed) % ship_num;
    swap(c.priority_gene[r1], c.priority_gene[r2]);
    for (int i = 0; i < mutation_num; i++) {
        int ship_id = rand() % ship_num;
        auto available = available_ports(ships[ship_id]);
        int r = rand() % available.size();
        c.port_gene[ship_id] = available[r];
    }
}

void gen_thread(int l,
                int r,
                vector<Chromosome>& children,
                Chromosome& best,
                vector<Chromosome>& population,
                function<pair<int, int>(unsigned int*)> select,
                unsigned int* seed,
                mutex& m) {
    for (int i = l; i < r; i++) {
        auto [parent1, parent2] = select(seed);
        children[i] = crossover(population[parent1], population[parent2], seed);
        mutation(children[i], seed);
        children[i].fitness = calc_fitness(children[i]);
        m.lock();
        if (children[i].fitness < best.fitness) {
            best = children[i];
        }
        m.unlock();
    }
}

Chromosome genetic(vector<Chromosome>& population) {
    auto start_time = chrono::steady_clock::now();
    Chromosome best;
    best.fitness = 1e9;
    signal(SIGINT, [](int) { break_tag = true; });
    unsigned int global_seeds[thread_num];
    for (int i = 0; i < thread_num; i++) {
        global_seeds[i] = rand();
    }
    vector<Chromosome> children(population_size);
    for (int gen = 0; gen < max_generation; gen++) {
        Chromosome cur_best;
        cur_best.fitness = 1e9;
        for (int i = 0; i < population_size; i++) {
            if (population[i].fitness < cur_best.fitness) {
                cur_best = population[i];
            }
        }
        if (gen * population_size % 100000 == 0 && gen != 0) {
            auto end_time = chrono::steady_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
            cerr << "generation: " << gen
                 << "\tcur best: " << cur_best.fitness
                 << "\tbest: " << best.fitness
                 << "\tspeed: " << gen * population_size / duration.count() * 1000 << " it/s" << endl;
        }
        // roulette wheel selection
        vector<double> prob(population_size);
        function<pair<int, int>(unsigned int*)> select;

        if (selection_method == SelectionMethod::Roulette) {
            for (int i = 0; i < population_size; i++) {
                prob[i] = quick_power((i + 1) / population_size, tournament_size);
                cerr << prob[i] << endl;
            }
            auto roulette_wheel_selection =
                [&prob](unsigned int* seed) {
                    double r1 = (double)rand_r(seed) / RAND_MAX;
                    double r2 = (double)rand_r(seed) / RAND_MAX;
                    int parent1 = lower_bound(prob.begin(), prob.end(), r1) - prob.begin();
                    int parent2 = lower_bound(prob.begin(), prob.end(), r2) - prob.begin();
                    return make_pair(parent1, parent2);
                };
            select = roulette_wheel_selection;
        } else if (selection_method == SelectionMethod::Tournament) {
            auto tournament_selection =
                [&population](unsigned int* seed) {
                    int parent1 = rand_r(seed) % population_size;
                    int parent2 = rand_r(seed) % population_size;
                    for (int i = 1; i < tournament_size; i++) {
                        int t = rand() % population_size;
                        if (population[t].fitness < population[parent1].fitness) {
                            parent1 = t;
                        }
                        t = rand() % population_size;
                        if (population[t].fitness < population[parent2].fitness) {
                            parent2 = t;
                        }
                    }
                    return make_pair(parent1, parent2);
                };
            select = tournament_selection;
        } else if (selection_method == SelectionMethod::Truncation) {
            nth_element(population.begin(), population.begin() + tournament_size, population.end());
            auto truncation_selection =
                [&population](unsigned int* seed) {
                    int parent1 = rand_r(seed) % truncation_size;
                    int parent2 = rand_r(seed) % truncation_size;
                    return make_pair(parent1, parent2);
                };
            select = truncation_selection;
        }

        mutex m;
        vector<thread> pool;
        int step = (population_size + thread_num - 1) / thread_num;
        int l = 0, r = step;
        for (int t_id = 0; t_id < thread_num; t_id++) {
            pool.emplace_back(
                gen_thread,
                l, r, ref(children), ref(best), ref(population), select, &global_seeds[t_id], ref(m));
            l += step, r += step;
            if (r > population_size)
                r = population_size;
        }
        for (int t_id = 0; t_id < thread_num; t_id++) {
            pool[t_id].join();
        }
        copy(children.begin(), children.end(), population.begin());
        if (break_tag) {
            break;
        }
    }
    signal(SIGINT, SIG_DFL);
    sort(population.begin(), population.end());
    if (population[0].fitness < best.fitness) {
        best = population[0];
    }
    return best;
}

int main(int argc, char* argv[]) {
    parser.add_argument("-d", "--dataset")
        .help("ships<dataset>.txt")
        .default_value(string("160"));
    parser.add_argument("-o", "--output")
        .help("output directory")
        .default_value(string("./test"));
    parser.add_argument("-p", "--population")
        .help("population size")
        .default_value(population_size)
        .action([](const string& value) { population_size = stoi(value); });
    parser.add_argument("-g", "--generation")
        .help("generation size")
        .default_value(max_generation)
        .action([](const string& value) { max_generation = stoi(value); });
    parser.add_argument("--toursize")
        .help("tournament size")
        .default_value(tournament_size)
        .action([](const string& value) { tournament_size = stoi(value); });
    parser.add_argument("--truncrate")
        .help("truncation rate")
        .default_value(truncation_rate)
        .action([](const string& value) { truncation_rate = stod(value); });
    parser.add_argument("-c", "--cpnum")
        .help("cross point num")
        .default_value(cross_point_num)
        .action([](const string& value) { cross_point_num = stoi(value); });
    parser.add_argument("-m", "--mutation")
        .help("mutation rate")
        .default_value(mutation_rate)
        .action([](const string& value) { mutation_rate = stod(value); });
    parser.add_argument("-t", "--thread")
        .help("thread num")
        .default_value(thread_num)
        .action([](const string& value) { thread_num = stoi(value); });
    parser.add_argument("-s", "--selection")
        .help("selection method")
        .default_value(0)
        .action([](const string& value) { selection_method = (SelectionMethod)stoi(value); });
    parser.add_argument("-r", "--round")
        .help("round")
        .default_value(1)
        .action([](const string& value) { round_num = stoi(value); });
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
    ship_num = stoi(dataset);
    truncation_size = truncation_rate * population_size;
    ships.resize(ship_num);
    ports.resize(port_num);
    FILE* fp = fopen(dataset_path.c_str(), "r");
    for (int i = 0; i < ship_num; i++) {
        int id;
        fscanf(fp, "%d,%d,%d,%d,%d", &id, &ships[i].arrival_time, &ships[i].work_time,
               &ships[i].length, &ships[i].depth);
    }
    fclose(fp);
    fp = fopen("../data/ports.txt", "r");
    for (int i = 0; i < port_num; i++) {
        int id;
        fscanf(fp, "%d,%d,%d", &id, &ports[i].length, &ports[i].depth);
    }

    available_ports_list.resize(ship_num);
    for (int i = 0; i < ship_num; i++) {
        available_ports_list[i] = available_ports(ships[i]);
    }

    // srand(time(NULL));
    srand(0);
    vector<int> priority_gene(ship_num);
    iota(priority_gene.begin(), priority_gene.end(), 0);
    vector<Chromosome> population(population_size);

    Chromosome best;
    for (int round_id = 0; round_id < round_num; round_id++) {
        for (int i = 0; i < population_size; i++) {
            random_shuffle(priority_gene.begin(), priority_gene.end());
            population[i].priority_gene = priority_gene;
            for (int j = 0; j < ship_num; j++) {
                int idx = rand() % available_ports_list[j].size();
                population[i].port_gene[j] = available_ports_list[j][idx];
            }
            population[i].fitness = calc_fitness(population[i]);
        }
        mutation_num = ship_num * mutation_rate;
        auto start = chrono::steady_clock::now();
        auto round_best = genetic(population);
        auto end = chrono::steady_clock::now();
        if (round_id == 0 || round_best.fitness < best.fitness) {
            best = round_best;
        }
        cerr << "time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
        cerr << "round:" << round_id << "\tbest fitness : " << round_best.fitness << endl;
        if (break_tag) {
            break;
        }
    }
    cerr << "best fitness: " << best.fitness << endl;

    cout << "dataset: " << dataset << endl;
    cout << "population: " << population_size << endl;
    cout << "generation: " << max_generation << endl;
    cout << "tournament size: " << tournament_size << endl;
    cout << "cross point num: " << cross_point_num << endl;
    cout << "mutation rate: " << mutation_rate << endl;
    cout << "thread num: " << thread_num << endl;
    cout << "selection method: " << ((selection_method == SelectionMethod::Tournament) ? "tournament" : "roulette") << endl;
    cout << "round: " << round_num << endl;
    cout << "best fitness: " << best.fitness << endl;
    output(best, output_path);
}