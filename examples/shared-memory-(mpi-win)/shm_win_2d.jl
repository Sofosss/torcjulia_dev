"""
Shared Distributed Memory (SDM) for 2D Arrays Example

Demonstrates torcjulia's shared memory capabilities for 2D arrays

Workflow:
1. Allocate a shared 2D array A of size (Nrow x Ncol) where Nrow = number of processes
2. Master process (rank 0) initializes A
3. Each process i is assigned to update row i of the shared array
4. All processes access the shared 2D array and modify their assigned row

Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 2 shm_win_2d.jl
"""

using torcjulia


function update_row(A, row)
    Ncol = size(A, 2)
    old_vals = A[row, :]
    
    for j in 1:Ncol
        A[row, j] += 10 * row + j
    end
    
    new_vals = A[row, :]
    println("MPI process $(torcjulia.node_id()): Row $row updated from $old_vals to $new_vals")
end

function main()
    Nrow = Int(torcjulia.num_nodes())
    Ncol = 3
    
    println("allocating shared 2D array of size ($Nrow x $Ncol) across $Nrow MPI processes")
    A = torcjulia.shm_alloc((Nrow, Ncol), Int)

    if torcjulia.node_id() == 0
        for i in 1:Nrow, j in 1:Ncol
            A[i, j] = 100 * (i - 1) + j
        end
        println("initial 2D shared array A: $A")
    end

    for i in 1:Nrow
        torcjulia.submit(update_row, A, i; qid = i-1)
    end

    torcjulia.wait()

    println("updated 2D A: $A")
end

torcjulia.init(main)