"""
Work Stealing Example

Demonstrates torcjulia's work stealing mechanism with imbalanced workloads (workloads with irregular parallelism) across MPI nodes.

Setup:
- Even-ranked nodes (0, 2, 4, ...): Execute tasks with 0.2s delay (artificial bottleneck)
- Odd-ranked nodes (1, 3, 5, ...): Execute tasks immediately

Scenario 1 - Work Stealing Enabled:
Tasks are distributed via round-robin. Idle workers can steal pending tasks from overloaded nodes, achieving dynamic (adaptive) load balancing

Scenario 2 - Work Stealing Disabled:
Tasks are distributed via round-robin but work stealing is disabled, so tasks remain on their initially assigned nodes without dynamic redistribution. Therefore, the bottleneck from tasks 
assigned to even-ranked nodes becomes more pronounced, resulting in longer overall execution time

Requirements: 2+ MPI processes, 1 worker thread per process

Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 2 work_stealing.jl
"""

using torcjulia


function work(x)
    torcjulia.node_id() % 2 == 0 ? sleep(0.2) : nothing
    y = x * x
    println("taskfun inp = $x, out = $y ...on node $(torcjulia.node_id())")
    y
end

function main()
    ntasks = 40
    torcjulia.enable_stealing()

    start_time = torcjulia.gettime()
    
    tasks = []
    for i in 1:ntasks
        push!(tasks, torcjulia.submit(work, i))
    end

    torcjulia.wait()
    
    end_time = torcjulia.gettime()

    for task in tasks
        println("received: $(torcjulia.input(task))^2=$(torcjulia.result(task))")
    end

    println("elapsed time (1st scenario) = $(round(end_time - start_time, digits = 5)) seconds\n")

    torcjulia.disable_stealing()

    start_time = time()
    
    tasks = []
    for i in 1:ntasks
        push!(tasks, torcjulia.submit(work, i))
    end

    torcjulia.wait()
    
    end_time = time()

    for task in tasks
        println("received: $(torcjulia.input(task))^2=$(torcjulia.result(task))")
    end

    println("elapsed time (2nd scenario) = $(round(end_time - start_time, digits = 5)) seconds")
end

nodes = torcjulia.num_nodes()
if nodes < 2
    println("[rank: $(torcjulia.node_id())] this example needs at least two Julia processes. exiting...")
    exit()
end

local_workers = torcjulia.num_local_workers()
if local_workers > 1
    println("[rank: $(torcjulia.node_id())] this example should use one worker thread per MPI process. exiting...")
    exit()
end

torcjulia.init(main)