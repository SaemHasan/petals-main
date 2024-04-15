import numpy as np
from pymoo.core.problem import Problem

# variables: number of blocks, number of servers
# objectives: latency, throughput
# latency of a block is the sum of the RTT of the servers that hold the block
# throughput of a block is the minimum throughput of the servers that hold the block
# For instance, let's say we have M servers and N blocks. 
# Our matrix representation (denoted as x) would be an M*N matrix, where each element x[i][j] represents whether server i is utilized for training/inference in block j
# The objective function would be to minimize the sum of latencies and maximize the sum of throughputs across all blocks.
# The constraints would be that each block must be assigned to at least one server.
# The decision variables would be the assignment of servers to blocks, and the objective function would be the sum of latencies and throughputs across all blocks.

class ChainSequence(Problem):
    def __init__(self, n_blocks, m_servers, rtts, throughputs):
        self.rtts = rtts
        self.throughputs = throughputs
        self.n_blocks = n_blocks
        self.m_servers = m_servers
        self.x = np.zeros((m_servers, n_blocks))
        super().__init__(n_var=2, n_obj=2, n_constr=1, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        latency = 0
        throughput = 0
        for i in range(self.m_servers):
            for j in range(self.n_blocks):
                latency += x[i][j] * self.rtts[i][j]
                throughput += x[i][j] * self.throughputs[i][j]
        throughput = 1 / throughput
        out["F"] = np.column_stack([latency, throughput])
        out["G"] = np.column_stack([np.sum(x, axis=0) - 1])
