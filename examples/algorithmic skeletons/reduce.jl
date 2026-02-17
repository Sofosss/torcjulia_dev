"""
Parallel Reduce (Skeleton) Example

Demonstrates torcjulia's `reduce` skeleton for aggregation operations on matrices and vectors

The parallel `reduce` skeleton:
- applies an associative reduction operation across matrix or vector elements
- supports different reduction directions -> :row or :column (for matrices)
- supports two modes:
  * "1d": reduces to a single value per row/column → vector result (or scalar for vectors)
  * "2d": reduces progressively: matrix → vector → scalar ("hierarchical" reduction)

Direction options (for matrices):
- :column → reduces along columns (aggregates rows): (mxn) → (1xn) or scalar
- :row → reduces along rows (aggregates columns): (mxn) → (mx1) or scalar

For vectors: direction is ignored, reduces to a single scalar value

Matrix (1d mode): reduce(+, A, direction = :column) → [col1_sum, col2_sum, col3_sum, ...]
Matrix (1d mode): reduce(+, A, direction = :row) → [row1_sum, row2_sum, row3_sum, ...]
Matrix (2d mode): reduce(+, A) → matrix → vector → scalar
Vector: reduce(+, [1,2,3,4,5]) → 15

Run: mpiexecjl -n 8 julia --project=/path/to/torcjulia/project --threads 3 reduce.jl
"""

include(joinpath(@__DIR__, "..", "..", "src", "torcjulia.jl"))
import .torcjulia

using Random


Random.seed!(1234)
   
@inline function work(a::Int, b::Int, c::Int)::Int
    a + b + c
end

function main()
    vec = rand(1:100, 50_000); mat = rand(1:15, 1_000, 100)

    println("vector size: $(length(vec))")
    println("matrix size: $(size(mat))")

    t0 = torcjulia.gettime() 
    res_vec = torcjulia.reduce(
        work,
        vec;
        args = (10,),
        chunksize = 1_000
    )
    elapsed = torcjulia.gettime() - t0
    println("result: $res_vec")
    println("expected result: $(Base.reduce((a, b) -> work(a, b, 10), vec))")
    println("elapsed time: $(round(elapsed, digits = 5)) seconds")

    t0 = torcjulia.gettime() 
    res1d_col = torcjulia.reduce(
        work,
        mat;
        args = (5,),
        chunksize = 500,
        direction = :column
    )
    elapsed = torcjulia.gettime() - t0
    println("result shape: $(size(res1d_col))")
    println("result (first 5 elements): $(res1d_col[1:5])")
    println("correct results? $(all(res1d_col .== [Base.reduce((a, b) -> work(a, b, 5), mat[:, i]) for i in 1:size(mat, 2)]))")
    println("elapsed time: $(round(elapsed, digits = 5)) seconds")

    t0 = torcjulia.gettime() 
    res1d_row = torcjulia.reduce(
        work,
        mat;
        args = (-2,),
        chunksize = 20,
        direction = :row
    )
    elapsed = torcjulia.gettime() - t0
    
    println("result shape: $(size(res1d_row))")
    println("(result (first 5 elements): $(res1d_row[1:5])")
    println("correct result? $(all(res1d_row .== [Base.reduce((a, b) -> work(a, b, -2), mat[i, :]) for i in 1:size(mat, 1)]))")
    println("elapsed time: $(round(elapsed, digits = 5)) seconds")

    t0 = torcjulia.gettime() 
    res2d = torcjulia.reduce(
        work,
        mat;
        args = (8,),
        chunksize = 500,
        mode = "2d"
    )
    elapsed = torcjulia.gettime() - t0 

    println("result: $res2d")
    println("expected result: $(Base.reduce((a, b) -> work(a, b, 8), [Base.reduce((a, b) -> work(a, b, 8), mat[:, i]) for i in 1:size(mat, 2)]))")
    println("elapsed time: $(round(elapsed, digits = 5)) seconds")
end

torcjulia.start(main)