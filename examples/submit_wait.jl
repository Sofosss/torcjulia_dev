"""
Baseline Task Submission Example

Demonstrates the typical submit-wait pattern in torcjulia

Notes:
- the main thread on the master node (rank 0) submits tasks via `submit()`
- tasks are distributed across available workers using round-robin scheduling
- the `wait()` call blocks the master thread until all submitted tasks complete
- while waiting, the main thread (of the master node) participates in executing tasks rather than remaining idle

Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 2 submit_wait.jl
"""

include(joinpath(@__DIR__, "..", "src", "torcjulia.jl"))
import .torcjulia


function work(x)
    x^2
end

function main()
    n_tasks = 10
    data = 1:n_tasks
    
    println("submitting $n_tasks tasks")
    
    futures = [torcjulia.submit(work, x) for x in data]

    torcjulia.wait(futures)

    for (i, fut) in enumerate(futures)
        res = torcjulia.result(fut)
        println("$(data[i])² = $res")
    end
end

torcjulia.start(main)