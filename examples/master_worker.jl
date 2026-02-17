"""
Master-Worker Pattern Example

Demonstrates the fundamental master-worker paradigm in torcjulia. The master process 
(rank 0) creates and submits tasks, then waits for their completion

Key feature: while waiting, the master's main thread actively participates in executing tasks rather than remaining idle

The master node submits eight tasks that compute squares with an artificial one-second delay, 
distributing them among the available workers in a round-robin fashion

Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 2 master_worker.jl
"""

include(joinpath(@__DIR__, "..", "src", "torcjulia.jl"))
import .torcjulia


function work(x)
    sleep(1)
    y = x^2

    println("work inp = $(round(x, digits = 3)), out = $(round(y, digits = 3)) \
             on node $(torcjulia.node_id()) worker $(torcjulia.worker_id())")
    y
end

function main()
    ntasks = 8
    inputs = 1:ntasks

    println("submitting $ntasks tasks")

    t0 = torcjulia.gettime()

    tasks = [torcjulia.submit(work, x) for x in inputs]
    torcjulia.wait(tasks)

    elapsed = torcjulia.gettime() - t0
    
    for (i, task) in enumerate(tasks)
        x = inputs[i]
        y = torcjulia.result(task)
        println("res: $x^2 = $y")
    end

    println("elapsed time: $(round(elapsed, digits = 5)) seconds")
end


torcjulia.start(main)