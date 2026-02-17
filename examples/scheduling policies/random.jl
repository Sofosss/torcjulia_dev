"""
Random Scheduling Example

Demonstrates torcjulia's Random scheduling policy, where each task is assigned to a randomly selected node at runtime

The scheduling policy is set via `set_scheduling_policy(:RANDOM)`

Unlike the other supported scheduling policies, task distribution is non-deterministic
Each task is independently assigned to a random node, so the load balance may vary between runs

Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 2 random.jl
"""

include(joinpath(@__DIR__, "..", "..", "src", "torcjulia.jl"))
import .torcjulia


@inline function work(x::Int)::Int
    sleep(rand())
    y = x^2
    println("work inp = $(round(x, digits = 3)), out = $(round(y, digits = 3)) \
             on node $(torcjulia.node_id()) -> worker $(torcjulia.worker_id())")

    y
end

function main()
    ntasks = 14; inputs = 1:ntasks

    torcjulia.set_scheduling_policy(:RANDOM)

    tasks = [torcjulia.submit(work, x) for x in inputs]
    torcjulia.wait(tasks)

    for (i, task) in enumerate(tasks)
        x = inputs[i]
        y = torcjulia.result(task)
        println("result[$i]: $x^2 = $y")
    end
end

torcjulia.start(main)