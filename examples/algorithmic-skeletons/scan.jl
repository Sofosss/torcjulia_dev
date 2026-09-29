"""
Parallel Scan (Skeleton) Example

Demonstrates torcjulia's `scan` skeleton for prefix sum operations

The parallel `scan` skeleton:
- computes cumulative results of an associative operation
- supports two modes:
  * "inclusive": includes current element → [a₁, a₁+a₂, a₁+a₂+a₃, ...]
  * "exclusive": excludes current element → [0, a₁, a₁+a₂, a₁+a₂+a₃, ...]

inclusive scan: scan(+, [1,2,3,4]) → [1, 3, 6, 10]
exclusive scan: scan(+, [1,2,3,4]) → [0, 1, 3, 6]

Run: mpiexecjl -n 8 julia --project=/path/to/torcjulia/project --threads 3 scan.jl
"""

using torcjulia


@inline function work(x::Int, y::Int)::Int
    x + y
end

function main()
    N = 2_000; chunksize = 100
    a = collect(Int64, 1:N)

    println("chunksize: $chunksize elements per task")
    println("expected tasks: $(div(N, chunksize))")

    t0 = torcjulia.gettime()
    result_excl = torcjulia.scan(work, a; chunksize = chunksize, mode = "exclusive")
    elapsed = torcjulia.gettime() - t0
    
    expected_excl = zeros(Int64, N)
    for i in 2:N
        expected_excl[i] = Base.reduce((x, y) -> work(x, y), a[1:i-1])
    end
    
    println("result (first 10 elements): $(result_excl[1:10])")
    println("correct results: $(all(result_excl .== expected_excl))")
    println("elapsed time: $(round(elapsed, digits = 5)) seconds")

    t0 = torcjulia.gettime()
    result_incl = torcjulia.scan(work, a; chunksize = chunksize, mode = "inclusive")
    elapsed = torcjulia.gettime() - t0
    
    expected_incl = similar(a, Int64)
    for i in 1:N
        expected_incl[i] = Base.reduce((x, y) -> work(x, y), a[1:i])
    end
    
    println("result (first 10 elements): $(result_incl[1:10])")
    println("correct results? $(all(result_incl .== expected_incl))")
    println("elapsed time: $(round(elapsed, digits = 5)) seconds")
end

torcjulia.init(main)