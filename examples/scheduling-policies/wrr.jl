"""
Weighted Round-Robin (WRR) Scheduling Example

Demonstrates torcjulia's WRR scheduling policy, where tasks are distributed across nodes proportionally based on user-defined weights 
Nodes with higher weights receive more tasks

The scheduling policy is set via `set_scheduling_policy(:WRR, weights)`, which propagates the policy and weights to all MPI nodes

With weights = [1, 1, 2, 3] (total = 7) and 14 tasks across 4 nodes (1 worker thread each), tasks are assigned in cycles of 7:

Cycle 1 (tasks  1- 7)        Cycle 2 (tasks  8-14)
─────────────────────        ─────────────────────
node 0 (w=1) → [1]           node 0 (w=1) → [8]
node 1 (w=1) → [2]           node 1 (w=1) → [9]
node 2 (w=2) → [3, 4]        node 2 (w=2) → [10, 11]
node 3 (w=3) → [5, 6, 7]     node 3 (w=3) → [12, 13, 14]

Summary:
┌────────┬────────┬────────────────────────┬─────────┬──────┐
│  Node  │ Weight │          Tasks         │  Count  │   %  │
├────────┼────────┼────────────────────────┼─────────┼──────┤
│ node 0 │   1    │ [1, 8]                 │    2    │  14% │
│ node 1 │   1    │ [2, 9]                 │    2    │  14% │
│ node 2 │   2    │ [3, 4, 10, 11]         │    4    │  29% │
│ node 3 │   3    │ [5, 6, 7, 12, 13, 14]  │    6    │  43% │
└────────┴────────┴────────────────────────┴─────────┴──────┘

Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 2 wrr.jl
"""

using torcjulia


@inline function work(x::Int)::Int
    sleep(rand())
    y = x^2
    println("work inp = $(round(x, digits = 3)), out = $(round(y, digits = 3)) \
             on node $(torcjulia.node_id()) -> worker $(torcjulia.worker_id())")
    
    y
end

function main()
    ntasks = 14; inputs = 1:ntasks
    weights = [1, 1, 2, 3]

    torcjulia.set_scheduling_policy(:WRR,  weights)

    tasks = [torcjulia.submit(work, x) for x in inputs]
    torcjulia.wait(tasks)

    for (i, task) in enumerate(tasks)
        x = inputs[i]; y = torcjulia.result(task)
        println("result[$i]: $x^2 = $y")
    end
end

torcjulia.init(main)