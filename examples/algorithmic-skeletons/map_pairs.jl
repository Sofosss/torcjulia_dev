"""
Parallel MapPairs Paradigm (Skeleton) Example

Demonstrates torcjulia's `mapPairs` skeleton 
This pattern applies a function to every (x, y) pair from Cartesian product of two input arrays, producing a result matrix

`mapPairs((x, y) -> x + y, [1, 2], [10, 20])` → `[11 21; 12 22]`

Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 3 map_pairs.jl
"""

using torcjulia


@inline function work(x::Int, y::Int)::Float64
    s::Float64 = 0.0
    @inbounds @simd for i in 1:5000
        s += sin(x * i) * cos(y * i)
    end
    s
end

function main()
    a = 1:1000; b = 100:-1:1
    chunksize = 1000
    
    println("total pairs: $(length(a) * length(b))")
    println("expected tasks: $(div(1000 * 100, chunksize))")

    t0 = torcjulia.gettime()
    results = torcjulia.mapPairs(work, a, b; chunksize = chunksize)
    elapsed = torcjulia.gettime() - t0
    
    println("result matrix size: $(size(results))")
    println("correct results? $(results == ([work(x, y) for x in a, y in b]))")
    println("elapsed time: $(round(elapsed, digits = 5)) seconds")    
end

torcjulia.init(main)