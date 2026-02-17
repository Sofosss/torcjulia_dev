"""
Multi-Level Task Dependencies with Nested Parallelism Example

Similar to the multilevel_dependencies.jl example, but demonstrates task dependencies combined with nested parallelism
A parent task (task_4) spawns child tasks that form their own dependency chain

Generated DAG:
```
task_1 (2s) ──> task_2 (1s) ──┐
                              ├──> task_4 ──> [task_5 (1.2s) ──┐
task_3 (1.5s) ────────────────┘                                ├──> task_7 (1s) ──> task_8 (0.7s)
                                               task_6 (0.5s) ──┘
```

Key features:
- task dependencies at the top level -> task_2 depends on task_1
- nested parallelism -> task_4 creates its own sub-DAG of tasks
- automatic synchronization at both levels (main tasks and nested tasks)

Expected execution flow:
1. task_1 and task_3 start immediately (independent, run in parallel)
2. task_2 waits for task_1 to complete
3. task_4 waits for both task_2 and task_3
4. task_4 spawns 4 nested tasks with their own dependency chain:
   - task_5 and task_6 run in parallel
   - task_7 waits for both task_5 and task_6
   - task_8 waits for task_7

Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3 nested_dependencies.jl
"""

include(joinpath(@__DIR__, "..", "..", "src", "torcjulia.jl"))
import .torcjulia


function task_1()
    sleep(2)
    result = 10
    println("[task 1] completed → returning $result")
    result
end

function task_2(res)
    res_val = torcjulia.result(res)
    result = res_val * 2
    println("[task 2] received $res_val from task 1 → returning $result")
    sleep(1)
    result
end

function task_3()
    sleep(1.5)
    result = 5
    println("[task 3] completed → returning $result")
    result
end

function task_4(res2, res3)
    r2 = torcjulia.result(res2)
    r3 = torcjulia.result(res3)
    println("[task 4] received $r2 from task 2 and $r3 from task 3")

    fut_5 = torcjulia.submit(task_5, r2)     # 20 + 1 → 21
    fut_6 = torcjulia.submit(task_6, r3)     # 5 × 3 → 15
    fut_7 = torcjulia.submit(task_7, fut_5, fut_6)  # 21 + 15 → 36
    fut_8 = torcjulia.submit(task_8, fut_7)  # 36² → 1296

    torcjulia.wait([fut_8])

    result = torcjulia.result(fut_8)
    
    println("[task 4] nested tasks completed → result: $result")
end

function task_5(val)
    result = val + 1
    println("[task 5] processing $val → returning $result")
    sleep(1.2)
    result
end

function task_6(val)
    result = val * 3
    println("[task 6] processing $val → returning $result")
    sleep(0.5)
    result
end

function task_7(fut1, fut2)
    r1 = torcjulia.result(fut1)
    r2 = torcjulia.result(fut2)
    result = r1 + r2
    println("[task 7] received $r1 from task 5 and $r2 from task 6 → returning $(r1 + r2)")
    sleep(1)
    result
end

function task_8(fut)
    res = torcjulia.result(fut)
    result = res^2
    println("[task 8] processing ($res)² → returning $result")
    sleep(0.7)
    result
end

function main()
    torcjulia.enable_stealing()

    fut_1 = torcjulia.submit(task_1)           # → 10
    fut_2 = torcjulia.submit(task_2, fut_1)    # 10 × 2 → 20
    fut_3 = torcjulia.submit(task_3)           # → 5
    _ = torcjulia.submit(task_4, fut_2, fut_3) 

    torcjulia.wait()
end

torcjulia.start(main)