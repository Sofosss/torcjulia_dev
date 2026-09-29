"""
Parallel Filter (Skeleton) Example

Demonstrates torcjulia's `filter` skeleton

The parallel `filter` skeleton:
- applies a user-defined function to each element of an iterable
- returns only elements where the provided function returns true
- automatically partitions the data into chunks for parallel processing
- creates a task for each chunk, distributed across available workers
- preserves the original order of filtered elements

`filter(work, [1,2,3,4,5,6])` → `[2,4,6]`

Run: mpiexecjl -n 8 julia --project=/path/to/torcjulia/project --threads 3 filter.jl
"""

using torcjulia


@inline function work(x::Int)::Bool
    sleep(0.004)
    x % 2 == 0
end

function main()
    N = 10_000; chunksize = 100
    data = 1:N

    println("chunksize: $chunksize elements per task")
    println("expected tasks: $(div(N, chunksize))")

    t0 = torcjulia.gettime()
    filtered_data = torcjulia.filter(work, data; chunksize = chunksize)
    elapsed = torcjulia.gettime() - t0

    println("filtered results: $(length(filtered_data)) elements")
    println("correct results? $(all(x -> x % 2 == 0, filtered_data))")
    println("elapsed time: $(round(elapsed, digits = 5)) seconds")
end


torcjulia.init(main)