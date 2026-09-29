"""
Task Priority Example with Work Stealing

Demonstrates torcjulia's priority-based task scheduling. The runtime maintains separate 
queues for different priority levels, and work stealing 
prioritizes high-priority tasks when redistributing work across nodes

Priority levels:
- `TaskPriority.high`: urgent tasks (green output)
- `TaskPriority.medium`: medium priority tasks (yellow output)
- `TaskPriority.low`: low priority tasks (cyan output, default)

Scenario:
- 15 general-purpose tasks (default low priority, cyan output) are submitted first
- 25 priority tasks with random priorities (high/medium/low) are submitted afterward
- work stealing is enabled, allowing idle workers to steal pending tasks
- when stealing, high-priority tasks are preferentially stolen over lower-priority ones

Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 3 priority_work_stealing.jl
"""

using torcjulia

using Crayons, Random


function priority_task_func(i, p)
    color = p == torcjulia.TaskPriority.high ? :green :
            p == torcjulia.TaskPriority.medium ? :yellow : :cyan
    
    println(Crayon(foreground = color)("priority task $i [$(p.level)] started"))
    
    sleep(0.6 + rand() * 0.2)
    
    println(Crayon(foreground = color)("priority task $i [$(p.level)] finished"))
end

function generic_task_func(i)
    println(Crayon(foreground = :cyan)("generic task $i [low] started"))
    
    sleep(0.8 + rand() * 0.2)
    
    println(Crayon(foreground = :cyan)("generic task $i [low] finished"))
end


function main()
    Random.seed!(150)
    
    num_generic_tasks = 15; num_priority_tasks = 25

    torcjulia.enable_stealing()

    priorities = [torcjulia.TaskPriority.low, torcjulia.TaskPriority.medium, torcjulia.TaskPriority.high]

    println("submitting $num_generic_tasks generic tasks (default low priority)")
    for i in 1:num_generic_tasks
        torcjulia.submit(generic_task_func, i)
    end

    println("submitting $num_priority_tasks tasks with random priorities")
    for (i, p) in enumerate(rand(priorities, num_priority_tasks))
        torcjulia.submit(priority_task_func, i, p; priority = p)
    end

    torcjulia.wait()
    
    println(Crayon(foreground = :magenta)("all tasks completed ✓"))
end

torcjulia.init(main)