"""
Shared Distributed Memory (SDM) Example

Demonstrates torcjulia's shared memory capabilities using MPI windows
Multiple MPI processes on the same compute node can access and modify a shared array concurrently,
with each process updating a different portion of the array

Workflow:
1. allocate a shared array A of size N (where N = number of MPI processes)
2. master process (rank 0) initializes A with values [0, 100, 200, ...]
3. each process i is assigned a task to update A[i] by adding 99*i
4. all processes access the shared array and modify their assigned element

Notes:
- `shm_alloc()` -> creates MPI window-based shared memory accessible across processes on the same node
- multiple processes can read/write the shared array

Run: mpiexecjl -n 8 julia --project=/path/to/torcjulia/project --threads 2 shm_win.jl
"""

include(joinpath(@__DIR__, "..", "..", "src", "torcjulia.jl"))
import .torcjulia


function work(idx; A)
    old_val = A[idx]
    A[idx] += 99 * idx
    new_val = A[idx]
    println("MPI process $(torcjulia.node_id()): A[$idx] updated from $old_val to $new_val")
end

function main()
    N = torcjulia.num_nodes()
    
    println("allocating shared array of size $N across $N MPI processes")
    A = torcjulia.shm_alloc((Int(N),), Int)

    if torcjulia.node_id() == 0
        for i in 1:N
            A[i] = 100 * (i - 1)
        end
        println("initial shared array A: $A")
    end

    for i in 1:Int(N)
        torcjulia.submit(work, i; A = A, qid = i-1)
    end

    torcjulia.wait()

    println("updated A: $A")
end

torcjulia.start(main)