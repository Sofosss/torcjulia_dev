"""
Parallel Starmap (Skeleton) Example

Demonstrates torcjulia's `starmap` skeleton

Key difference from `map`:
- `map(f, [(1,2), (3,4)])` → `[f((1,2)), f((3,4))]` (passes tuple as single arg)
- `starmap(f, [(1,2), (3,4)])` → `[f(1,2), f(3,4)]` (unpacks tuple elements as separate args)

Run: mpiexecjl -n 8 julia --project=/path/to/torcjulia/project --threads 3 starmap.jl
"""

include(joinpath(@__DIR__, "..", "..", "src", "torcjulia.jl"))
import .torcjulia


@inline function work(x::Int, y::Int, mul::Int; param::Int = 0)::Int
    sleep(0.004)
    (x + y + param) * mul
end

function main()
    N = 10_000; chunksize = 100
    data = [(i, i * 2) for i in 1:N]
    param = 4; mul = 2

    println("chunksize: $chunksize (tuple) elements per task")
    println("expected tasks: $(div(N, chunksize))")

    t0 = torcjulia.gettime()
    results = torcjulia.starmap(work, data; chunksize = chunksize, args = (mul,), param = param)
    elapsed = torcjulia.gettime() - t0

    expected = [(x + y + param) * mul for (x, y) in data]
    println("correct results? $(all(results .== expected))")
    println("elapsed time: $(round(elapsed, digits = 5)) seconds")
end

torcjulia.start(main)