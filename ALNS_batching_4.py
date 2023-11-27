# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 12:31:48 2022

@author: User7
"""

import copy
import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy.random as rnd
import pandas as pd
import random
import time

from alns import ALNS, State
from alns.accept import *
from alns.stop import *
from alns.weights import *
import csv

class TspState(State):
    """
    Solution class for the TSP problem. It has two data members, nodes, and edges.
    nodes is a list of node tuples: (id, coords). The edges data member, then, is
    a mapping from each node to their only outgoing node.
    """

    def __init__(self, nodes):
        self.nodes = nodes
        
    def objective(self):
        """
        The objective function is simply the sum of all individual edge lengths,
        using the rounded Euclidean norm.
        """
        return calculateObj(self.nodes, OP)
    
def calculateObj(solution, op):
    cur_fit = 0
    cur_fit2 = 0
    timewindow_len = noBatches + noZones - 1
    timewindow = [None]*timewindow_len
    
    for t in range(timewindow_len):
        timewindow[t] = [] 
        
    batch_seq = [None]*noBatches
    for b in range(noBatches):
        batch_seq[b] = [0]*noZones
    
    batch_count = 0
    batch_index = 0
    timewindow_index = 0
    for o in range(noOrders):
        t = 0
        for s in solution:
            if int(s)==o+1:
#                 for z in range(noZones):
#                     timewindow[timewindow_index+z].append(OP[t][z])
# #                     print(OP[t][z], t, z)
                for z in range(noZones):
                    batch_seq[batch_index][z] += OP[t][z]

                batch_count += 1
                if batch_count == Capa:
                    batch_count = 0
                    for z in range(noZones):
                        timewindow[batch_index+z].append(batch_seq[batch_index][z])
                    batch_index += 1
            t += 1

        timewindow_index += 1
        
    
    for t in range(len(LastBatch)):
        timewindow[t].append(LastBatch[t])
        
    for t in range(timewindow_len):
        cur_fit += (max(timewindow[t]))*PT
#         print(max(timewindow[t]))
#     print(timewindow)
    return cur_fit

def random_repair(current, rnd_state):
    """
    Greedily repairs a tour, stitching up nodes that are not departed
    with those not visited.
    """
    destroyed_order=[]

    destroyed_seq = []
    for o in range(noOrders):
        destroyed_seq.append(o+1)


    for o in range(len(current.nodes)):
        if current.nodes[o] == 0:
            destroyed_order.append(o+1)
        else:
            destroyed_seq.remove(current.nodes[o])

    # This kind of randomness ensures we do not cycle between the same
    # destroy and repair steps every time.
    shuffled_idcs = rnd_state.permutation(destroyed_seq)

    for s in range(len(shuffled_idcs)):
        current.nodes[destroyed_order[s]-1] = shuffled_idcs[s]

    return current

def greedy_repair(current, rnd_state):
    """
    Greedily repairs a tour, stitching up nodes that are not departed
    with those not visited.
    """
    destroyed_order=[]

    destroyed_seq = []
    for o in range(noOrders):
        destroyed_seq.append(o+1)


    for o in range(len(current.nodes)):
        if current.nodes[o] == 0:
            destroyed_order.append(o+1)
        else:
            destroyed_seq.remove(current.nodes[o])

    # This kind of randomness ensures we do not cycle between the same
    # destroy and repair steps every time.
    shuffled_idcs = rnd_state.permutation(destroyed_seq)

    for s in shuffled_idcs:

        if len(destroyed_order) != 1:
            best_fit1 = 999999
            best_fit2 = 0
            loc1, loc2 = -1, -1
            
            for d in destroyed_order:
                destroyed = copy.deepcopy(current)
                destroyed.nodes[d-1] = s
                
                seed_batch = destroyed.nodes.index(s)//Capa #batch index of seed order

                pre_batch = seed_batch -1
                suc_batch = seed_batch + 1

                segment = []
                
                if pre_batch >= 0:
                    segment.append(destroyed.nodes[pre_batch * Capa : (pre_batch+1) * (Capa)]) #input the preceding batch

                segment.append(destroyed.nodes[seed_batch * Capa : (seed_batch+1) * (Capa)]) # input the seed batch

                if suc_batch <= noBatches-1:
                    segment.append(destroyed.nodes[suc_batch * Capa : (suc_batch+1) * (Capa)]) #input the succedding batch
                
                cur_fit1, cur_fit2 = seg_calculateObj(segment) # calculate local objval

                if best_fit1 > cur_fit1: # minimize maximum workloads
                    best_fit1 = cur_fit1
                    best_fit2 = cur_fit2
                    order_to_assign = d


                elif best_fit1 == cur_fit1 and best_fit2 < cur_fit2: # tie break (maximize minimum workloads)
                    best_fit1 = cur_fit1
                    best_fit2 = cur_fit2
                    order_to_assign = d
                
            destroyed_order.remove(order_to_assign)
            current.nodes[order_to_assign-1] = s
        else:
#             print(destroyed_order)
            current.nodes[destroyed_order[0]-1] = s

    return current

def seg_calculateObj(segment):
    segment_workloads = []
    for s in segment:
        workloads = [0] * noZones
        for i in range(Capa):
            for z in range(noZones):
                workloads[z] = workloads[z] + OP[s[i]-1][z]
        segment_workloads.append(workloads)

    timewindow = [None] * (len(segment_workloads) + noZones - 1)
    for t in range(len(segment_workloads) + noZones - 1):
        timewindow[t] = []

    for o in range(len(segment_workloads)):
        for z in range(noZones):
            timewindow[o + z].append(segment_workloads[o][z])

    cur_fit = 0
    cur_fit2 = 0

    for t in range(len(timewindow)):
        if timewindow[t]:
            cur_fit += (max(timewindow[t]))*PT
            cur_fit2 += (min(timewindow[t]))*PT
        
    return cur_fit, cur_fit2

def random_removal(current, rnd_state):
    """
    Random removal iteratively removes random edges.
    """
    destroyed = copy.deepcopy(current)

    for idx in rnd_state.choice(len(destroyed.nodes),edges_to_remove(destroyed),replace=False):
        destroyed.nodes[idx] = 0
    return destroyed

def worst_removal(current, rnd_state):
    destroyed = copy.deepcopy(current)
    batch_list = [i for i in range(noBatches)]

    worst_batches = sorted(batch_list,key=lambda seq: local_objVal(seq, init_sol))

    selected_batches= []
    for idx in range(edges_to_remove(init_sol)):
        selected_batches.append(worst_batches[-(idx + 1)])
    
    for i in selected_batches:
        destroyed.nodes[destroyed.nodes.index(random.sample(destroyed.nodes[i * Capa : (i + 1) * Capa], 1)[0])]=0

    return destroyed


def local_objVal(seed_batch, destroyed):

    if seed_batch == 0:
        adj_batch = 1
    elif seed_batch == noBatches - 1:
        adj_batch = noBatches - 2
    else: 
        r = random.random()
        if r > 0.5:
            adj_batch = seed_batch + 1
        else:
            adj_batch = seed_batch - 1
        
    segment = []
    
    if adj_batch > seed_batch:
        segment.append(destroyed.nodes[seed_batch * Capa : (seed_batch+1) * (Capa)]) # input the seed batch
        segment.append(destroyed.nodes[adj_batch * Capa : (adj_batch+1) * (Capa)]) #input the preceding batch
    else:
        segment.append(destroyed.nodes[adj_batch * Capa : (adj_batch+1) * (Capa)]) #input the preceding batch
        segment.append(destroyed.nodes[seed_batch * Capa : (seed_batch+1) * (Capa)]) # input the seed batch


    cur_fit1, cur_fit2 = seg_calculateObj(segment) # calculate local objval

    return cur_fit1


def edges_to_remove(state):
    return int(len(state.nodes) * degree_of_destruction)

file_directory = ""
read = open(file_directory + "tftoCPlex.csv", "r", encoding='UTF-8')
tftoCPlex = pd.DataFrame(list(csv.reader(read, delimiter='\t')))
noOrders = int(tftoCPlex[0][0])
noZones = int(tftoCPlex[0][1])
noBatches = int(tftoCPlex[0][3])
Capa = int(tftoCPlex[0][2])
batchNo = int(tftoCPlex[0][4])
DemandCharacteristic = int(tftoCPlex[0][5])

# RunTime =  int(float(tftoCPlex[1][6]))
destroy_rate =  (float(tftoCPlex[1][7]))
Iteration = int(float(tftoCPlex[1][6]))
# RunTime = 60

PT = int(float(tftoCPlex[1][1]))
LastBatch = []
for z in range(noZones):
    LastBatch.append(int(tftoCPlex[0][z+7]))

OP = [[0]*noZones for _ in range(noOrders)]
for o in range(noOrders):
    for z in range(noZones):
        
        OP[o][z] = int(tftoCPlex[z+3][o])
        
initial_solution_1 = []
for o in range(noOrders):
    initial_solution_1.append(int(tftoCPlex[2][o]))
        
read.close()

SEED = 43
random_state = rnd.RandomState(SEED)

start = time.time()

init_sol = TspState(initial_solution_1)


degree_of_destruction = destroy_rate

alns = ALNS(random_state)

alns.add_destroy_operator(random_removal)
alns.add_destroy_operator(worst_removal)

alns.add_repair_operator(greedy_repair)
alns.add_repair_operator(random_repair)

crit = SimulatedAnnealing(start_temperature=1_000,
                          end_temperature=1,
                          step=1 - 1e-3,
                          method="exponential")
weight_scheme = SimpleWeights([3, 2, 1, 0.5], 2, 2, 0.8)
# stop = MaxRuntime(RunTime)
stop = MaxIterations(Iteration)

result = alns.iterate(init_sol, weight_scheme, crit, stop)
SolvingTime = round(time.time() - start,2)

solution = result.best_state
bestObj = solution.objective()

# pct_difference = 100 * (objective - optimal) / optimal

# print(f"Best heuristic objective is {objective}.")
# print(f"This is {pct_difference:.1f}% worse than the optimal solution, which is {optimal}.")

# _, ax = plt.subplots(figsize=(12, 6))
# result.plot_objectives(ax=ax, lw=2)



#result output
if batchNo == 1:
    f1 = open(file_directory + 'Result_SA2_'+str(noZones)+'z_'+str(DemandCharacteristic)+'variable_' +str(int(noOrders/noBatches))+ 'BatchWindow'+'.csv','w')
    writer = csv.writer(f1, delimiter=';', lineterminator='\n')
    writer.writerow([batchNo, bestObj, SolvingTime])
    f1.close()
else:
    f1 = open(file_directory + 'Result_SA2_'+str(noZones)+'z_'+str(DemandCharacteristic)+'variable_' +str(int(noOrders/noBatches))+ 'BatchWindow'+'.csv','a')
    writer = csv.writer(f1, delimiter=';', lineterminator='\n')
    writer.writerow([batchNo, bestObj, SolvingTime])
    f1.close()


orderseq = solution.nodes
pd.DataFrame({'seq':orderseq}).to_csv(file_directory + 'order_seq.csv', sep=';')