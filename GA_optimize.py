import os
import random
import multiprocessing
import time
import numpy as np
from deap import base, creator, tools, algorithms

import sys
sys.path.append('./')
from analysis.NN_train import RegressionOpti
from analysis.feature_extraction import FeatureExtractor


name1 = "ZCH"
name2 = "LZ"
file_dir = "./"
p_1 = []
p_2 = []
files = os.listdir(file_dir)

for filename in files:
    if filename.startswith(f"{name1}_") and filename.endswith(".npy"):
        filepath = os.path.join(file_dir, filename)
        data = np.load(filepath)
        p_1.append(data)

for filename in files:
    if filename.startswith(f"{name2}_") and filename.endswith(".npy"):
        filepath = os.path.join(file_dir, filename)
        data = np.load(filepath)
        p_2.append(data)
r_opti = RegressionOpti(8,6)
featureExtractor = FeatureExtractor(p_1,p_2,cached=True)

def get_effectiveness(combination):
    # get_effectiveness.calls += 1
    # start_time = time.time()
    psd =  featureExtractor.get_PSD(combination)
    plv = featureExtractor.get_PLV(combination)
    y = featureExtractor.get_Y()
    data = [(psd[i],plv[i],y[i]) for i in range(len(y))]
    result = r_opti.train_eval(data)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # get_effectiveness.total_time += elapsed_time
    # average_time = get_effectiveness.total_time / get_effectiveness.calls
    # print(f"'get_effectiveness' Avg execution time: {average_time:.6f} s      ", end='\r')
    return result

# get_effectiveness.calls = 0
# get_effectiveness.total_time = 0

# 参数设置
NUM_OBJECTIVES = 32
NUM_SELECTED = 8

# 定义问题
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# 定义如何随机选择目标
toolbox.register("attribute", random.randint, 0, NUM_OBJECTIVES - 1)
def unique_individual_creation():
    sample = random.sample(range(NUM_OBJECTIVES), NUM_SELECTED)
    sorted_sample = sorted(sample)
    return creator.Individual(sorted_sample)
toolbox.register("individual", unique_individual_creation)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 评估函数
def evalEffectiveness(individual):
    return get_effectiveness(individual),

toolbox.register("evaluate", evalEffectiveness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=NUM_OBJECTIVES-1, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# 并行设置
def setup_pool(pool_size):
    pool = multiprocessing.Pool(processes=pool_size)
    toolbox.register("map", pool.map)

import matplotlib.pyplot as plt

def plot_progress(log):
    gen = log.select("gen")
    fit_mins = log.select("min")
    fit_avgs = log.select("avg")
    fit_maxs = log.select("max")

    plt.ion()  # 打开交互模式
    plt.plot(gen, fit_mins, label="Minimum Fitness")
    plt.plot(gen, fit_avgs, label="Average Fitness")
    plt.plot(gen, fit_maxs, label="Maximum Fitness")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Progress")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.ioff()  # 关闭交互模式


def main(pool_size=4):
    setup_pool(pool_size)
    print("----GA began running----\n")
    random.seed(64)
    pop = toolbox.population(n=512)

    # 指定一些初始组合
    initial_combinations = [[random.randint(0, NUM_OBJECTIVES-1) for _ in range(NUM_SELECTED)] for _ in range(10)]
    for comb in initial_combinations:
        pop.append(creator.Individual(comb))

    # 遗传算法参数
    CXPB, MUTPB, NGEN = 0.7, 0.5, 80

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats=stats, verbose=True)
    
    best_ind = tools.selBest(pop, 1)[0]
    print('----GA finished----')
    print("Best Individual = ", best_ind, "\nBest Fitness = ", best_ind.fitness.values)

    print('----full log----')
    for ind in pop:
        print("Individual = ", ind, "\nFitness = ", ind.fitness.values)

    try:
        plot_progress(log)
    except Exception:
        print('plot_progress error')
        

if __name__ == "__main__":
    main()

