"""
Local Scheduling Example

Demonstrates torcjulia's Local scheduling policy, where tasks are always assigned to the node that created them

The scheduling policy is set via `set_scheduling_policy(:LOCAL)`

With 12 tasks across 4 nodes (1 worker thread each), since all tasks are submitted by the master node (rank 0), all tasks are assigned to node 0

Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 2 local.jl
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

    torcjulia.set_scheduling_policy(:LOCAL)

    tasks = [torcjulia.submit(work, x) for x in inputs]
    torcjulia.wait(tasks)

    for (i, task) in enumerate(tasks)
        x = inputs[i]
        y = torcjulia.result(task)
        println("result[$i]: $x^2 = $y")
    end
end

torcjulia.init(main)