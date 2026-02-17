"""
Parallel MapReduce Paradigm (Skeleton) Example

Demonstrates torcjulia's `mapReduce` skeleton, implementing the classic MapReduce paradigm 
This pattern combines a map phase (applying a function to each element) with a reduce phase (aggregating results into a scalar value)

Key parameters:
- `map_func`: transformation applied to each element
- `reduce_func`: binary associative operation for aggregation (e.g., +, *, max, min)

Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 3 map_reduce.jl
"""

include(joinpath(@__DIR__, "..", "..", "src", "torcjulia.jl"))
import .torcjulia


@inline function work(x::Int)::Int
    sleep(0.002)
    x ^ 2
end

@inline function aggregate(x::Int, y::Int)::Int
    sleep(0.002)
    x + y
end

function main()
    N = 5_000; chunksize = 250
    data = 1:N

    println("chunksize: $chunksize elements per task")
    println("tasks: $(div(N, chunksize))")

    t0 = torcjulia.gettime()
    result = torcjulia.mapReduce(work, aggregate, data; chunksize = chunksize)
    elapsed = torcjulia.gettime() - t0

    println("result: $result")
    println("expected result: $(sum(x -> x * x, data))")
    println("elapsed time: $(round(elapsed, digits = 5)) seconds")
end

torcjulia.start(main)