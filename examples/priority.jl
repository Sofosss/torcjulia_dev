"""
Task Priority Scheduling Example

Demonstrates torcjulia's priority-based task execution. Tasks are scheduled according 
to their priority level, with higher-priority tasks executing before lower-priority ones 
when workers become available

Priority levels:
- `TaskPriority.high`: urgent tasks (green output)
- `TaskPriority.medium`: medium priority tasks (yellow output)
- `TaskPriority.low`: low priority tasks (cyan output, default)

Scenario:
- 8 low-priority tasks are submitted first
- 8 medium-priority tasks are submitted second
- 8 high-priority tasks are submitted last

Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 2 priority.jl
"""

include(joinpath(@__DIR__, "..", "src", "torcjulia.jl"))
import .torcjulia

using Crayons


const PRIORITY_COLOR = Dict(
    :low    => :cyan,
    :medium => :yellow,
    :high   => :green
)

function work(i, p)
    cr = Crayon(foreground = PRIORITY_COLOR[p.level])
    sleep(0.5)
    println(cr("task $i finished (priority = $(p.level))"))
end

function main()
    println("submitting 24 tasks: 8 low, 8 medium, 8 high priority")
    
    priorities = vcat(
        fill(torcjulia.TaskPriority.low, 8),
        fill(torcjulia.TaskPriority.medium, 8),
        fill(torcjulia.TaskPriority.high, 8)
    )

    for (i, p) in enumerate(priorities)
        torcjulia.submit(work, i, p; priority = p)
    end

    torcjulia.wait()
    
    println(Crayon(foreground = :magenta)("all tasks completed ✓"))
end

torcjulia.start(main)