"""
Nested Parallelism Example - Fibonacci

Demonstrates torcjulia's support for multi-level nested parallelism via recursive
Fibonacci computation. Tasks dynamically spawn child tasks, forming arbitrary-depth
task hierarchies.

The implementation uses a hybrid approach to control tasks granularity:
- small subproblems (n < 30): computed sequentially to avoid task overhead
- large subproblems (n ≥ 30): Recursively decomposed into parallel subtasks

Well-suited for:
- divide-and-conquer algorithms
- recursive computations with independent subproblems
- adaptive algorithms that decide parallelization granularity at runtime

Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 3 nested_tasks.jl
"""

using torcjulia


function fib(n::UInt64)
    if n == 0
        return 0
    elseif n == 1
        return 1
    else
        n_1 = n - 1
        n_2 = n - 2
        
        if n < 30
            result1 = fib(n_1)
            result2 = fib(n_2)
            result = result1 + result2
        else
            task1 = torcjulia.submit(fib, n_1)
            task2 = torcjulia.submit(fib, n_2)
            
            torcjulia.wait(; return_completed = false)
            
            result = torcjulia.result(task1) + torcjulia.result(task2)
        end
        
        result
    end
end

function main()
    n = UInt64(43)
    
    start_time = torcjulia.gettime()
    result = fib(n)
    end_time = torcjulia.gettime()
    
    elapsed = round(end_time - start_time, digits = 5)
    
    println("result -> fib($n)=$result")
    println("elapsed time: $elapsed seconds")
end

torcjulia.init(main)