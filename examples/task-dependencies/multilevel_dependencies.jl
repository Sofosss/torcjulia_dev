"""
Multi-Level Task Dependencies Example

Demonstrates building a multi-level task dependency graph (DAG) in torcjulia
Tasks can depend on other tasks through both positional and keyword arguments, creating layered data flow patterns

Generated DAG:
```
    task_1 (2s) ──┐
                  ├──> task_3 (1s) ──> task_4 (1s) ──┐
                                                       ├──> task_6 (1s) ──> final result
    task_2 (1.5s) ──────────────────> task_5 (0.5s) ─┘
```

Notes:
- multi-level dependencies -> task_3 depends on task_1, task_4 depends on task_3, etc.
- mixed dependency specification -> positional args (e.g., task_3) and keyword args (e.g., task_4, task_5)
- task_1 and task_2 run concurrently, as do task_4 and task_5
- the runtime ensures execution order respects all dependencies

Expected execution flow:
1. task_1 and task_2 start immediately (independent, run in parallel)
2. task_3 waits for task_1 to complete
3. task_4 waits for task_3; task_5 waits for task_2 (run in parallel)
4. task_6 waits for both task_4 and task_5

Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 2 multilevel_dependencies.jl
"""

using torcjulia

using Crayons


function task_1()
    sleep(2)
    println(Crayon(foreground = :yellow)("[task 1] completed -> returning 10"))
    return 10
end

function task_2()
    sleep(1.5)
    println(Crayon(foreground = :yellow)("[task 2] completed -> returning 5"))
    5
end

function task_3(res)
    val = torcjulia.result(res)
    println(Crayon(foreground = :cyan)("[task 3] received $val from task 1 -> returning $(val * 2)"))
    sleep(1)
    val * 2
end

function task_4(; fut)
    val = torcjulia.result(fut)
    sleep(1)
    println(Crayon(foreground = :green)("[task 4] received $val from task 3 → returning $(val + 1)"))
    val + 1
end

function task_5(; fut)
    val = torcjulia.result(fut)
    sleep(0.5)
    println(Crayon(foreground = :green)("[task 5] received $val from task 2 → returning $(val * 3)"))
    val * 3
end

function task_6(; futures::Tuple)
    val1 = torcjulia.result(futures[1])
    val2 = torcjulia.result(futures[2])
    sleep(1)
    println(Crayon(foreground = :red)("[task 6] received $val1 from task 4 and $val2 from task 5 → returning $(val1 + val2)"))
    val1 + val2
end

function main()
    torcjulia.enable_stealing()

    res_1 = torcjulia.submit(task_1) # 10      
    res_2 = torcjulia.submit(task_2) # 5        

    res_3 = torcjulia.submit(task_3, res_1)  # 10 * 2 = 20 

    res_4 = torcjulia.submit(task_4; fut = res_3)  # 20 + 1 = 21
    res_5 = torcjulia.submit(task_5; fut = res_2)  # 5 * 3 = 15

    res_6 = torcjulia.submit(task_6; futures = (res_4, res_5))  # 21 + 15 = 36

    torcjulia.wait()

   println(Crayon(foreground = :magenta)("result: $(torcjulia.result(res_6))"))
end

torcjulia.init(main)