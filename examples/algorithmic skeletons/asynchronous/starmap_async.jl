"""
Parallel Asynchronous Starmap (Skeleton) Example

Demonstrates torcjulia's `starmap_async` skeleton, the asynchronous variant of `starmap`
Similar to `map_async` but designed for iterables of tuples, where each tuple is unpacked as positional arguments to the user-provided function

The key difference from `map_async`:
- `map_async(f, [a, b, c])` → calls f(a), f(b), f(c)
- `starmap_async(f, [(a1,a2), (b1,b2)])` → calls f(a1,a2), f(b1,b2)

Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3 starmap_async.jl
"""

include(joinpath(@__DIR__, "..", "..", "..", "src", "torcjulia.jl"))
import .torcjulia


function work(x::Int, y::Int; scale::Float64 = 1.0, verbose::Bool = true)::Float64
    sleep(rand())
    result = (x^2 + y^2) * scale
    if verbose
        println("[work] -> ($x, $y) - result: $result")
    end
    result
end

function aggregate(a::Float64, b::Float64; verbose::Bool = true)::Float64
    sleep(0.2)
    result = a + b
    if verbose
        println("[aggregate] -> ($a, $b) - result: $result")
    end
    result
end

function main()
    torcjulia.enable_stealing()

    batch1  = [(1, 2), (3, 4), (5, 6), (7, 8)]
    batch2 = [(10, 20), (30, 40), (50, 60)]

    fut1 = torcjulia.starmap_async(work, batch1;
        callback = r -> println("[callback] -> batch 1 complete - result: $r"))

    fut2 = torcjulia.starmap_async(work, batch2; scale = 2.0,
        callback = r -> println("[callback] -> batch 2 complete - result: $r"))

    !torcjulia.ready(fut1) ? println("waiting for batch 1...") : nothing
    results1 = torcjulia.get(fut1)

    !torcjulia.ready(fut2) ? println("waiting for batch 2...") : nothing
    results2 = torcjulia.get(fut2)

    pairs = collect(zip(results1, results2))
    fut3 = torcjulia.starmap_async(aggregate, pairs;
        callback = r -> println("[callback] -> aggregation complete - result: $r"))

    results3 = torcjulia.get(fut3); total_score = sum(results3)
    println("final result: $total_score")

    expected_score = sum(
                        aggregate(a, b; verbose = false) 
                        for (a, b) in zip(
                            (work(x, y; verbose = false) for (x, y) in batch1),         
                            (work(x, y; scale = 2.0, verbose = false) for (x, y) in batch2) 
                        )
                    )
    println("expected result: $expected_score")
end


torcjulia.start(main)