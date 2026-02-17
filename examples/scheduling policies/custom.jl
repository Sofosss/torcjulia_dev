"""
Custom Scheduling Function Example

Demonstrates torcjulia's support for user-defined scheduling functions
Instead of using a built-in policy, the user can provide a custom function that returns the target node index for each task submission

The custom function is passed directly to `submit()` via the `distribution_policy` argument:
    torcjulia.submit(work, x; distribution_policy = my_scheduler)

The function must return a valid node index (0 to num_nodes-1) 
If an invalid value is returned, torcjulia falls back to Round-Robin

In this example, assuming 4 nodes, `scheduler()` routes tasks based on input parity: 
even tasks go to nodes 0 and 1, odd tasks go to nodes 2 and 3

With 12 tasks:

even tasks (2,4,6,...)    odd tasks (1,3,5,...)
──────────────────────    ─────────────────────
node 0 → [2, 6, 10]       node 2 → [1, 5, 9]
node 1 → [4, 8, 12]       node 3 → [3, 7, 11]

Summary:
┌────────┬──────────────┬─────────┬──────┐
│  Node  │    Tasks     │  Count  │   %  │
├────────┼──────────────┼─────────┼──────┤
│ node 0 │ [2, 6, 10]   │    3    │  25% │
│ node 1 │ [4, 8, 12]   │    3    │  25% │
│ node 2 │ [1, 5, 9]    │    3    │  25% │
│ node 3 │ [3, 7, 11]   │    3    │  25% │
└────────┴──────────────┴─────────┴──────┘

Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 2 custom.jl
"""

include(joinpath(@__DIR__, "..", "..", "src", "torcjulia.jl"))
import .torcjulia


@inline function work(x::Int)::Int
    sleep(0.2)
    y = x^2
    println("work inp = $(round(x, digits = 3)), out = $(round(y, digits = 3)) \
             on node $(torcjulia.node_id()) worker $(torcjulia.worker_id())")
    y
end

const _even_counter = Ref(0)
const _odd_counter  = Ref(0)
const _tasks_counter = Ref(0)

function scheduler()::Int
    if _tasks_counter[] % 2 == 0
        _even_counter[] += 1
        (_even_counter[] - 1) % 2      
    else
        _odd_counter[] += 1
        2 + (_odd_counter[] - 1) % 2  
    end
end

function main()
    ntasks = 12; inputs = 1:ntasks

    tasks::Vector{torcjulia.TorcTask} = []
    for (i, x) in enumerate(inputs)
        _tasks_counter[] += 1
        push!(tasks, torcjulia.submit(work, x; distribution_policy = scheduler))
    end
    
    torcjulia.wait(tasks)

    for (i, task) in enumerate(tasks)
        x = inputs[i]
        y = torcjulia.result(task)
        println("result[$i]: $x^2 = $y")
    end
end

torcjulia.start(main)