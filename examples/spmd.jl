"""
SPMD (Single Program Multiple Data) Example

Demonstrates torcjulia's SPMD execution model, where the same function runs on all MPI 
nodes (with node-specific data)

Scenario:
- executes the `work` function once per MPI node
- each node computes a local result based on its rank

Notes:
- all nodes execute the same code but operate on different data

Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 2 spmd.jl
"""

using torcjulia


function work(; multiplier::Int)
    node_id = torcjulia.node_id()
    A = [(node_id * 10 + i) * multiplier for i in 1:node_id]
    sum(A), node_id
end

function main()
    tasks = torcjulia.spmd(work; multiplier = 2)
    
    for task in tasks
        res = torcjulia.result(task)
        println("node $(res[2]) -> result = $(res[1])")
    end
end


torcjulia.init(main)