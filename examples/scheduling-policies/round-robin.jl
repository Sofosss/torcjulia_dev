"""
Round-Robin (RR) Scheduling Example

Demonstrates torcjulia's default RR scheduling policy, where tasks are distributed evenly across nodes in a cyclic manner

The scheduling policy is set via `set_scheduling_policy(:ROUND_ROBIN)`

Note: RR is the default task distribution policy in torcjulia

With 12 tasks across 4 nodes (1 worker thread each), tasks are assigned cyclically:

Cycle 1 (tasks 1-4)     Cycle 2 (tasks 5-8)     Cycle 3 (tasks 9-12)
────────────────────    ────────────────────    ────────────────────
node 0 → [1]            node 0 → [5]            node 0 → [ 9]
node 1 → [2]            node 1 → [6]            node 1 → [10]
node 2 → [3]            node 2 → [7]            node 2 → [11]
node 3 → [4]            node 3 → [8]            node 3 → [12]

Summary:
┌────────┬──────────────┬─────────┬──────┐
│  Node  │    Tasks     │  Count  │   %  │
├────────┼──────────────┼─────────┼──────┤
│ node 0 │ [1, 5,  9]   │    3    │  25% │
│ node 1 │ [2, 6, 10]   │    3    │  25% │
│ node 2 │ [3, 7, 11]   │    3    │  25% │
│ node 3 │ [4, 8, 12]   │    3    │  25% │
└────────┴──────────────┴─────────┴──────┘

Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 2 round-robin.jl
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
    ntasks = 12; inputs = 1:ntasks

    torcjulia.set_scheduling_policy(:ROUND_ROBIN)

    tasks = [torcjulia.submit(work, x) for x in inputs]
    torcjulia.wait(tasks)

    for (i, task) in enumerate(tasks)
        x = inputs[i]
        y = torcjulia.result(task)
        println("result[$i]: $x^2 = $y")
    end
end

torcjulia.init(main)