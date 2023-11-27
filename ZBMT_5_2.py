import time
import numpy as np
import math
from docplex.mp.model import Model
np.set_printoptions(threshold=np.inf)
import csv
import pandas as pd
#from plantsim.plantsim import Plantsim

manual_time_limit = 600
limit_gap = 0.000001

#model = Plantsim(version='16.0',license_type='research',visible=False)
#model.load_model('C:\\Users\\user\\Documents\\Huong\\OneDrive - pusan.ac.kr\\Huong\\2. DepoBatch\\2. Simulation\\RP-OBDS_export\\DepoBatch_sim_pickingtime_performance.spp')

    #ZBMX - JP edit

def ZBMX(noBatches, noOrders, noZones, Capa, op, time_limit_cplex, LastBatch, Pt):

    mdl = Model(name="ZBMX")

    orders = range(noOrders)
    batches = range(noBatches)
    zones = range(noZones)
    timewindows = range(noBatches + noZones - 1)
    zones_ = zones[:-1]

    Xoi = mdl.binary_var_matrix(orders,batches)
    P = mdl.continuous_var_matrix(batches, zones)
    # LastBatch[z] = mdl.continuous_var_list(zones)
    ST = mdl.continuous_var_list(batches)
    L = mdl.continuous_var_matrix(batches, zones)
    D = mdl.continuous_var_matrix(batches, zones)
    CP = mdl.continuous_var_matrix(batches, zones)
    CD = mdl.continuous_var_matrix(batches, zones)
    CW = mdl.continuous_var_matrix(batches, zones)
    S = mdl.continuous_var_matrix(batches, zones)
    # Pmax = mdl.continuous_var()
    

    mdl.add_constraints(mdl.sum(Xoi[o,i] for i in batches) == 1 for o in orders) #2
    mdl.add_constraints(mdl.sum(Xoi[o,i] for o in orders) <= Capa for i in batches) #3  
    mdl.add_constraints(P[i,z] == Pt * mdl.sum(Xoi[o,i] * op[o][z] for o in orders) for z in zones for i in batches) #4
    for i in batches:   #5
        if i == 1:
             mdl.add_constraints(ST[i]==0)
        else:
            mdl.add_constraints(ST[i]==L[i-1,1])
            
    for i in batches:   #6,7
        for z in zones_:
            if z == 1:
                 mdl.add_constraints(D[i,z]>=L[i-1,z+1]-CP[i,z]-CW[i,z])
            else:
                mdl.add_constraints(D[i,z]>=L[i-1,1]-CP[i,z]-CW[i,z]-CD[i,z-1])
            mdl.add_constraints(D[i,z]>=0)
                
    
    
            
    
            
    
   
    # mdl.add_constraints(CK[i+z] >= P[i,z] for z in zones for i in batches)
    # mdl.add_constraints(CK[z-1] >= Pt * LastBatch[z] for z in zones_)  #CKi+z-1 >= Piz i=0 (last batch)
    # mdl.add_constraints(Pmax >= CK[t] for t in timewindows) 

    # mdl.minimize(Pmax)
    mdl.minimize(mdl.sum(CK[t] for t in timewindows))

    if time_limit_cplex != 0: mdl.parameters.timelimit = time_limit_cplex
    if limit_gap != 0: mdl.parameters.mip.tolerances.mipgap.set(limit_gap)
    mdl.parameters.mip.cuts.cliques = -1
    mdl.parameters.mip.cuts.covers = -1
    mdl.parameters.mip.cuts.flowcovers = -1
    mdl.parameters.mip.cuts.implied = -1
    mdl.parameters.mip.cuts.gubcovers = -1
    mdl.parameters.mip.cuts.gomory = -1
    mdl.parameters.mip.cuts.pathcut = -1
    mdl.parameters.mip.cuts.mircut = -1
    mdl.parameters.mip.cuts.disjunctive = -1
    # no zerohalf cut limit
    mdl.parameters.mip.interval = 10
    mdl.parameters.emphasis.mip = 2 #CPX_MIPEMPHASIS_OPTIMALITY : Emphasize optimality over feasibility
    mdl.parameters.simplex.display = 2
    mdl.parameters.mip.display = 2
    
    
    if mdl.solve(clean_before_solve=True, log_output='ZBMT_solve_log_'+str(noZones)+'z_'+str(DemandCharacteristic)+'variable_' +str(int(noOrders))+ 'order''.txt'):
        objective = round(mdl.solution.get_objective_value())
        print("*** ZBMT Objective = ", objective)
        print(mdl.get_solve_status())
        print(mdl.get_solve_details())

        orderSequence = []

        for i in batches:
            count = 0
            for o in orders:
                if float(Xoi[o,i]) > 0.5:
                    orderSequence.append(o+1)
                    count = count + 1
                    if count == Capa:  #move on to the next batch if we find every orders in batch i
                        break
        
        
        if len(orderSequence) != len(orders):
            print("Order sequencing Error")
            return 0, orderSequence
        else:
            return objective, orderSequence       
        
    else:
        print("No solution")
        return 0 


file_directory = "C:\\Users\\User7\\Desktop\\JP\\ZBM\\"
read = open(file_directory + "tftoCPlex.csv", "r", encoding='UTF-8')
tftoCPlex = pd.DataFrame(list(csv.reader(read, delimiter='\t')))
noOrders = int(tftoCPlex[0][0])
noZones = int(tftoCPlex[0][1])
noBatches = int(tftoCPlex[0][3])
Capa = int(tftoCPlex[0][2])
batchNo = int(tftoCPlex[0][4])
DemandCharacteristic = int(tftoCPlex[0][5])

Pt = int(float(tftoCPlex[1][1]))
LastBatch = []
for z in range(noZones):
    LastBatch.append(int(tftoCPlex[0][z+7]))

op = [[0]*noZones for _ in range(noOrders)]
for o in range(noOrders):
    for z in range(noZones):
        
        op[o][z] = int(tftoCPlex[z+3][o])
        
read.close()

time_limit_cplex = manual_time_limit
    
        
# ------ MAIN PROCEDURE ------ #
start = time.time()

Objective, sequencelist = \
    ZBMX(noBatches, noOrders, noZones, Capa, op, time_limit_cplex, LastBatch, Pt)

SolvingTime = round(time.time() - start,2)

print(SolvingTime)
#result output
if batchNo == 1:
    f1 = open(file_directory + 'Result_ZBMT_'+str(noZones)+'z_'+str(DemandCharacteristic)+'variable_' +str(int(noBatches))+ 'BatchWindow'+'.csv','w')
    writer = csv.writer(f1, delimiter=';', lineterminator='\n')
    writer.writerow([batchNo, Objective, SolvingTime])
    f1.close()
else:
    f1 = open(file_directory + 'Result_ZBMT_'+str(noZones)+'z_'+str(DemandCharacteristic)+'variable_' +str(int(noBatches))+ 'BatchWindow'+'.csv','a')
    writer = csv.writer(f1, delimiter=';', lineterminator='\n')
    writer.writerow([batchNo, Objective, SolvingTime])
    f1.close()

orderseq = [0] * noOrders
for i in range(noOrders):
    s = 0
    while i+1 != sequencelist[s]:
        # print(sequencelist[s])
        s = s + 1
    
    if i+1 == sequencelist[s]:
        orderseq[i] = s+1

pd.DataFrame({'seq':orderseq}).to_csv(file_directory + 'order_seq.csv', sep=';')

f2 = open(file_directory + "Cplex_done.txt", "w")
f2.write("1")
f2.close()
