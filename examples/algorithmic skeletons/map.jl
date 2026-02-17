    """
Parallel Map (Skeleton) Example

Demonstrates torcjulia's `map` skeleton

The parallel `map` skeleton:
- applies a user-provided function to each element the provided iterable
- automatically partitions the data into chunks for parallel processing
- creates a task for each chunk, distributed across available workers
- returns results in the original input order

Key parameters:
- `chunksize`: number of elements per task (default: auto-calculated based on data size and worker count)

`map(f, [1,2,3])` → `[f(1), f(2), f(3)]`

Run: mpiexecjl -n 8 julia --project=/path/to/torcjulia/project --threads 3 map.jl
"""
    
    
include(joinpath(@__DIR__, "..", "..", "src", "torcjulia.jl"))
import .torcjulia


@inline function work(x::Int)::Int
    sleep(0.006)
    x^2
end

function main()
    N = 10_000; chunksize = 50
    data = 1:N

    println("chunksize: $chunksize elements per task")
    println("expected tasks: $(div(N, chunksize))")

    t0 = torcjulia.gettime()
    results = torcjulia.map(work, data; chunksize = chunksize)
    elapsed = torcjulia.gettime() - t0

    println("correct results? $(all(results .== data.^2))")
    println("elapsed time: $(round(elapsed, digits = 5)) seconds")
end

torcjulia.start(main)