"""
Parallel Asynchronous Map (Skeleton) Example

Demonstrates torcjulia's `map_async` skeleton, the asynchronous variant of `map`
Unlike `map` which blocks until all tasks complete, `map_async` returns immediately 
with an AsyncResult handle, allowing the caller to continue execution while tasks run in the background

AsyncResult provides three helper functions:
- `ready(ar)`: check if all nested tasks have completed (non-blocking)
- `wait(ar)`: block until all tasks complete
- `get(ar)`: block until completion and collect results into a flat vector

    map_async(f, data)
            |
            v
       AsyncResult
            |
            v
      .---------------.
      |               |
      v               v
   ready()        get() / wait()
 (non-blocking)        (blocking)
      |               |
      v               v
 true / false   results / nothing

`map_async` supports also an optional callback that executes asynchronously once all related tasks have completed, receiving the collected results as input

Execution Flow:

data = 1:N          vec = [...]        arr = [[...],[...],...]

   fetch_data           work1             work_batch
        |                |                   |
        v                v                   v
   fetch_fut          work1_fut         batch_fut
  (AsyncResult)      (AsyncResult)     (AsyncResult)
        |                |                   |
        v                v                   v
  ready() / get()    ready() / get()    ready() / get()
        |                |                   |
        v                v                   v
  fetched_data       work1_results       batch_results
        |                
        v                
     work2_fut           
        |                
        v
  ready() / get()
        |
        v
    work2_results
        |
        v
   agg_fut (aggregate)
        |
        v
   ready() / get()
        |
        v
    agg_results

total_score = agg_results["total"] + sum(batch_results) + sum(work1_results)

Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 5 map_async.jl
"""

using torcjulia


function fetch_data(id::Int; verbose::Bool = true)::Dict{String, Int}
    sleep(1.1)
    if verbose
        println("[fetch] -> ID $id")
    end

    Dict("id" => id, "value" => id * 3)
end

function work1(n::Int; verbose::Bool = true)::Int
    sleep(0.5)
    result = sum(i^2 for i in 1:n)
    if verbose
        println("[work1] -> n: $n - result: $result")
    end

    result
end

function work2(data::Dict{String, Int}; verbose::Bool = true)::Dict{String, Int}
    sleep(rand(0.3:0.1:0.8))
    result = data["value"] * 2 + 10
    if verbose
        println("[work2] -> ID $(data["id"]), value: $(data["value"]) - result: $result")
    end

    Dict("id" => data["id"], "processed_val" => result)
end

function aggregate(results::Vector{Dict{String, Int}}; verbose::Bool = true)::Dict{String, Float64}
    sleep(1.0)
    total = sum(r["processed_val"] for r in results)
    avg = total / length(results)
    if verbose
        println("[aggregate] -> total: $total, avg: $avg")
    end

    Dict("total" => total, "average" => avg)
end

function work_batch(batch::Vector{Int}; verbose::Bool = true)::Int
    sleep(0.5)
    result = sum(batch) * 2
    if verbose
        println("[work_batch] -> batch: $batch - result: $result")
    end

    result
end

function main()
    N = 4; data = 1:N; vec = [15, 30, 10]; arr = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]  
    
    torcjulia.enable_stealing()
    
    fetch_fut = torcjulia.map_async(fetch_data, data;
        callback = r -> println("[callback] -> fetch complete: $(length(r)) items"))
    
    work1_fut = torcjulia.map_async(work1, vec;
        callback = r -> println("[callback] -> work1 complete - result: $r"))
    
    !torcjulia.ready(fetch_fut) ? println("waiting for fetch results...") : nothing
    fetched_data = torcjulia.get(fetch_fut)
    
    work2_fut = torcjulia.map_async(work2, fetched_data; chunksize = 2,
        callback = r -> println("[callback] -> work2 complete: $(length(r)) items"))
    
    !torcjulia.ready(work1_fut) ? println("waiting for work1 results...") : nothing
    work1_results = torcjulia.get(work1_fut)

    !torcjulia.ready(work2_fut) ? println("waiting for work2 results...") : nothing
    work2_results = torcjulia.get(work2_fut)
    
    agg_fut = torcjulia.map_async(aggregate, [work2_results];
        callback = r -> println("[callback] -> aggregation complete - result: $(r[1]["average"]) (avg), $(r[1]["total"]) (total)"))
    
    batch_fut = torcjulia.map_async(work_batch, arr;
        callback = r -> println("[callback] -> batch processing complete - result: $r"))
    
    torcjulia.ready(agg_fut) ? println("waiting for aggregation results...") : nothing
    torcjulia.ready(batch_fut) ? println("waiting for batch processing results...") : nothing
    
    agg_results = torcjulia.get(agg_fut); batch_results = torcjulia.get(batch_fut)
    
    total_score = agg_results[1]["total"] + sum(batch_results) + sum(work1_results)
    
    println("final result: $total_score")

    expected_score = aggregate(
                        Base.map(x -> work2(x; verbose = false),
                            Base.map(y -> fetch_data(y; verbose = false), data));
                        verbose = false
                    )["total"] +
                    sum(Base.map(batch -> work_batch(batch; verbose = false), arr)) +
                    sum(Base.map(n -> work1(n; verbose = false), vec))
    println("expected result: $expected_score")
end

torcjulia.init(main)