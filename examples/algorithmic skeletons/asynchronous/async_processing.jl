"""
Asynchronous Processing Example

Combines `map_async` and `starmap_async` skeletons for concurrent asynchronous processing

(For details on each skeleton, see map_async.jl and starmap_async.jl)

Execution flow:

    pairs = [(1,10), (2,20), ...]           numbers = [1, 2, ..., 10]
           |                                         |
           v                                         v
    starmap_async(work1)                map_async(work2)
           |                                         |
           v                                         v
      fut1  <-- executed concurrently --> fut2
           |                                         |
           v                                         v
        get(fut1)                      get(fut2)
           |                                         |
           v                                         v
     results1                       results2
           |                                         |
           .-----------------.--------------------.
                             |
                             v
              map_async(aggregate, [results1, results2])
                             |
                             v
                        results3
                             |
                             v
              final_score = results3[1]["total"] + results3[2]["total"]

Run: mpiexecjl -n 2 julia --project=. --threads 4 async_processing.jl
"""

include(joinpath(@__DIR__, "..", "..", "..", "src", "torcjulia.jl"))
import .torcjulia


function work1(a::Int, b::Int; verbose::Bool = true)::Int
    sleep(1 + rand())
    result = a * 2 + b * 2
    if verbose
        println("[work1] -> ($a, $b) - result: $result")
    end
    
    result
end

function work2(x::Int; verbose::Bool = true)::Int
    sleep(2 + rand())
    result = x^2
    if verbose
        println("[work2] -> $x - result: $result")
    end
    
    result
end

function aggregate(values::Vector{Int}; verbose::Bool = true)::Dict{String, Float64}
    sleep(0.6)
    total = sum(values)
    avg   = total / length(values)
    if verbose
        println("[aggregate] total: $total, avg: $avg")
    end
    
    Dict("total" => total, "average" => avg)
end

function main()
    N = 16; pairs = [(i, i * N) for i in 1:N]; numbers = collect(1:N)

    fut1 = torcjulia.starmap_async(work1, pairs;
        callback = r -> println("[callback] starmap (work1) complete: $(length(r)) items"))

    fut2 = torcjulia.map_async(work2, numbers; chunksize = 4,
        callback = r -> println("[callback] map (work2) complete: $(length(r)) items"))

    results1 = torcjulia.get(fut1); results2  = torcjulia.get(fut2)

    fut3 = torcjulia.map_async(aggregate, [results1, results2];
        callback = r -> println("[callback] aggregation complete: $(length(r)) items"))

    results3 = torcjulia.get(fut3)

    total_score = results3[1]["total"] + results3[2]["total"]
    println("final result: $total_score")

    expected_score = sum(
        aggregate(vals; verbose = false)["total"]
        for vals in (
            [work1(a, b; verbose = false) for (a, b) in [(i, i*N) for i in 1:N]],
            [work2(x; verbose = false) for x in 1:N]
        )
    )
    println("expected result: $expected_score")
end

torcjulia.start(main)