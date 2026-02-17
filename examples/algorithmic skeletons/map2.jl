"""
Parallel Map (Skeleton) Example 2

Demonstrates torcjulia's `map` skeleton with multiple iterables

Unlike map.jl (single iterable), this example processes two iterables element-wise: `map(f, [1,2,3], [4,5,6])` → `[f(1,4), f(2,5), f(3,6)]`

Run: mpiexecjl -n 8 julia --project=/path/to/torcjulia/project --threads 3 map2.jl
"""

include(joinpath(@__DIR__, "..", "..", "src", "torcjulia.jl"))
import .torcjulia


@inline function work(x::Float64, y::Float64)::Float64
    sleep(0.005)
    x * y
end

function main()
    N = 10_000; chunksize = 100
    a = range(0, 2π, length = N); b = range(π, 3π, length = N)

    println("chunksize: $chunksize elements per task")
    println("expected tasks: $(div(N, chunksize))")

    t0 = torcjulia.gettime() 
    results = torcjulia.map(work, a, b; chunksize = chunksize)
    elapsed = torcjulia.gettime() - t0

    println("correct results? $(all(results .≈ a .* b))")
    println("elapsed time: $(round(elapsed, digits = 5)) seconds")
end


torcjulia.start(main)