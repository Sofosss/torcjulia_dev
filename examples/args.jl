"""
Positional and Keyword Arguments Example

Demonstrates how to pass both positional and keyword arguments to tasks in torcjulia.
Tasks are created with varying combinations of arguments, showcasing the flexibility 
of the task submission API.

Scenario:
- defines a work function accepting positional (a, b) and keyword (c, d) arguments
- submits multiple tasks with different argument (values) combinations
- displays input arguments and results for each task

Common use cases: parameter sweeps, hyperparameter tuning, and batch processing 

Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 4 args.jl
"""

using torcjulia

function _work(a, b; c = 0, d = 0)
    sleep(0.1 + 0.05 * rand())
    result = a + b + c + d
    println("worker on node $(torcjulia.node_id()): computing $a + $b + $c + $d = $result")
    result
end

function main()
    tasks = torcjulia.TorcTask[]
    
    for i in 1:4
        for j in 1:3
            pos_args = (i, j)
            kw_args = (; c = i * 10, d = j * 5)
            
            push!(tasks, torcjulia.submit(_work, pos_args...; kw_args...))
        end
    end

   println("submitted $(length(tasks)) tasks")

    for (idx, t) in enumerate(tasks)
        pos = torcjulia.input(t)
        kw = torcjulia.kw_input(t)
        pos_str = join(pos, ", ")
        kw_str = join(["$k=$v" for (k, v) in pairs(kw)], ", ")
        println("task $idx - positional: ($pos_str) - keyword: ($kw_str)")
    end

    torcjulia.wait(tasks)

    println("\nresults:")
    for (idx, t) in enumerate(tasks)
        pos = torcjulia.input(t)
        kw = torcjulia.kw_input(t)
        res = torcjulia.result(t)
        
        pos_str = join(pos, ", ")
        kw_str = join(["$k=$v" for (k, v) in pairs(kw)], ", ")
        println("task $idx: work($pos_str; $kw_str) = $res")
    end
end

torcjulia.init(main)