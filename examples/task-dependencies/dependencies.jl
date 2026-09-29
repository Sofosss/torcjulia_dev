"""
Task Dependencies Example

Demonstrates torcjulia's automatic handling of task dependencies. When a task depends on the results of other tasks, 
the runtime automatically manages execution order without explicit user-specified synchronization

Scenario:
- task A computes x² 
- task B computes x³
- task C combines results (x² + x³), and, thus, depends on both A and B 

Notes:
- dependencies are implicit: passing TorcTask objects as arguments automatically creates dependencies
- the runtime ensures dependent tasks execute only after their dependencies complete
- no manual/explicit wait() calls needed, since the scheduler handles execution order automatically
- calling `wait([taskC])` is sufficient; A and B complete automatically before C runs

Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 2 dependencies.jl
"""

using torcjulia


function workA(x)
    println("task A: computing x² where x = $x")
    x^2
end

function workB(x)
    println("task B: computing x³ where x = $x")
    x^3
end

function aggregate(a, b)
    result_a = torcjulia.result(a)
    result_b = torcjulia.result(b)
    println("task C: aggregating results from task A (x² = $result_a) and task B (x³ = $result_b)")
    result_a + result_b
end

function main()
    x = 5

    taskA = torcjulia.submit(workA, x)
    taskB = torcjulia.submit(workB, x)

    taskC = torcjulia.submit(aggregate, taskA, taskB)

    torcjulia.wait([taskC])

    println("[task A] x² = ", torcjulia.result(taskA))
    println("[task B] x³ = ", torcjulia.result(taskB))
    println("[task C] x² + x³ = ", torcjulia.result(taskC))
end

torcjulia.init(main)