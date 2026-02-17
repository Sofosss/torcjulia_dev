"""
Parallel MapPairsReduce Paradigm (Skeleton) Example

Demonstrates torcjulia's `mapPairsReduce` skeleton
This pattern computes a function for all (x, y) pairs from the Cartesian product of two input arrays, 
then aggregates the results into a vector or a scalar value

Key parameters:
- `map_func`: binary function applied to each (x, y) pair
- `reduce_func`: associative binary operation for aggregation (e.g., +, *, max, min)
- `mode`: "1d" for vector output, "2d" for scalar output
- `direction`: :row or :column (only for mode = "1d")

Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 3 map_pairs_reduce.jl
"""

include(joinpath(@__DIR__, "..", "..", "src", "torcjulia.jl"))
import .torcjulia


@inline function work(x::Int, y::Int)::Float64
    s::Float64 = 0.0
    @inbounds @simd for i in 1:1_000
        s += sin(x * i) * cos(y * i)
    end
    s
end

@inline function aggregate(x::Float64, y::Float64)::Float64
    max(x, y)
end

function main()
    a = 1:3000; b = 100:-1:1
    chunksize = 250
    
    println("chunksize: $chunksize elements per task")
    println("tasks: $(div(5000 * 100, chunksize))")

    t0 = torcjulia.gettime()
    result_1d_row = torcjulia.mapPairsReduce(work, aggregate, a, b; mode = "1d", direction=:row, chunksize = chunksize)
    elapsed = torcjulia.gettime() - t0

    println("result shape: $(size(result_1d_row))")
    println("elapsed time: $(round(elapsed, digits = 5)) seconds")

    t0 = torcjulia.gettime()
    result_1d_col = torcjulia.mapPairsReduce(work, aggregate, a, b; mode = "1d", direction=:column, chunksize = chunksize)
    elapsed = torcjulia.gettime() - t0

    println("result shape: $(size(result_1d_col))")
    println("elapsed time: $(round(elapsed, digits = 5)) seconds")

    t0 = torcjulia.gettime()
    result_2d = torcjulia.mapPairsReduce(work, aggregate, a, b; mode = "2d", chunksize = chunksize)
    elapsed = torcjulia.gettime() - t0

    println("result shap: $(length(result_2d))")
    println("elapsed time: $(round(elapsed, digits = 5)) seconds")
end


torcjulia.start(main)