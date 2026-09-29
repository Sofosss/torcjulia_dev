"""
Callback-Based Reduction Example

Demonstrates using callbacks for aggregating task results
Each task computes x² and its callback accumulates the result into a global sum

Notes:
- in torcjulia, callbacks are always executed on the workers of the node that created the task (the home node). Since all tasks are 
created by rank 0, all callbacks run on rank 0, ensuring safe access to the global `sum` variable
- if work stealing were enabled (disabled by default), callback tasks could be stolen and executed on remote nodes
where the global `sum` variable is not shared
- the global variable `sum` should be protected with a lock/semaphore for thread safety when 
multiple worker threads execute callbacks concurrently

Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 3 callback_reduction.jl
"""

using torcjulia


_sum = 0
sum_lock = ReentrantLock()  

function work(x)
    x^2
end

function cb(result)
    global _sum

    lock(sum_lock) do
        _sum += result
    end
end

function main()
    n = 20; data = 1:n
    
    println("computing sum of squares from 1 to $n using callbacks")
    
    tasks = [torcjulia.submit(work, x; callback = cb, async_callback = false) for x in data]
    torcjulia.wait(tasks)

    println("sum of squares from 1 to $n: $_sum")
    println("expected: $(sum(x^2 for x in 1:n))")
end

torcjulia.init(main)