"""
Task Pinning Example

Demonstrates torcjulia's task pinning feature

The `qid` parameter controls which node a task is assigned to
When work stealing is enabled, pinned tasks can still be stolen by idle nodes unless `forced_node = true` is set, 
which guarantees the task executes exclusively on the assigned node

Three scenarios (all tasks are submitted to node 0):

Scenario 1 - Pinned to node 0, stealing disabled:
  Tasks stay on node 0, other nodes remains idle

Scenario 2 - Pinned to node 0, stealing enabled:
  Other nodes steal pending tasks from node 0

Scenario 3 - Pinned to node 0, stealing enabled, forced_node = true:
  Tasks are guaranteed to execute on node 0, stealing is blocked

Requirements: 2+ MPI processes, 1 worker thread per process

Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 2 task_pinning.jl
"""

include(joinpath(@__DIR__, "..", "src", "torcjulia.jl"))
import .torcjulia


function work(x)
    torcjulia.node_id() % 2 == 0 ? sleep(0.2) : nothing
    y = x * x
    println("taskfun inp = $x, out = $y ...on node $(torcjulia.node_id())")

    y
end

function main()
    ntasks = 10
    
    println("work stealing: disabled")

    tasks = [torcjulia.submit(work, i; qid = 0) for i in 1:ntasks]
    torcjulia.wait()

    torcjulia.enable_stealing()
    println("\nwork stealing: enabled")

    tasks = [torcjulia.submit(work, i; qid = 0) for i in 1:ntasks]
    torcjulia.wait()

    println("\nwork stealing: enabled, forced_node: true")

    tasks = [torcjulia.submit(work, i; qid = 0, forced_node = true) for i in 1:ntasks]
    torcjulia.wait()
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

torcjulia.start(main)