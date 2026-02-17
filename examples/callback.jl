"""
Callback Functions Example

Demonstrates torcjulia's support for task callbacks. Each task can have an associated 
callback function that executes when the task completes

Notes:
- callbacks receive the task result as input
- `async_callback = false`: callbacks execute synchronously immediately after task completion
- `async_callback = true`: callbacks are queued as separate tasks for asynchronous execution

Tasks are assigned different callbacks based on their input value

Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3 callback.jl
"""

include(joinpath(@__DIR__, "..", "src", "torcjulia.jl"))
import .torcjulia


function cb_1(result)
    local_thread_id = torcjulia.worker_local_id()
    println("thread $local_thread_id on node $(torcjulia.node_id()): callback 1 -> result = $result")
end

function cb_2(result)
    local_thread_id = torcjulia.worker_local_id()
    println("thread $local_thread_id on node $(torcjulia.node_id()): callback 2 -> result = $result")
end

function work(x)
    y = x^2; local_thread_id = torcjulia.worker_local_id()
    println("thread $local_thread_id on node $(torcjulia.node_id()): work inp = $x, out = $y")
    y
end

function main()
    n_tasks = 16; data = 1:n_tasks

    tasks = [
        torcjulia.submit(work, x; 
            callback = x % 2 == 0 ? cb_1 : cb_2, 
            async_callback = false
        ) for x in data
    ]
    
    torcjulia.wait(tasks)

    for task in tasks
        res = torcjulia.result(task)
        println("received: $(torcjulia.input(task)[1])² = $res")
    end
end

torcjulia.start(main)