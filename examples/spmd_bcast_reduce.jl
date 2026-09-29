"""
SPMD with MPI Collective Operations Example

Demonstrates combining torcjulia's SPMD execution with MPI collective operations

Workflow:
1. master node (rank 0) initializes array A with values [0, 100, 200, ...]
2. all nodes replicate A via MPI.Bcast!
3. each node modifies A based on its rank (A *= rank)
4. sum all modified arrays using MPI.Allreduce

Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 2 spmd_bcast_reduce.jl
"""

using torcjulia

using MPI

const N = 16
const A = zeros(Int, N)

function bcast_task()
    comm = MPI.COMM_WORLD
    MPI.Bcast!(A, comm)
end

function local_computation()
    node_id = torcjulia.node_id()
    
    A .*= node_id
    
    println("node $node_id (after local computation): A = $A")
end

function reduce_task()
    node_id = torcjulia.node_id()

    comm = MPI.COMM_WORLD
    result = MPI.Allreduce(A, +, comm)

    println("node $node_id (after reduction): sum(A) = $result")
end

function main()
    if torcjulia.node_id() == 0
        for i in 1:N
            A[i] = 100 * (i - 1)
        end
        println("node 0 (initialization): A = $A")
    end

    torcjulia.spmd(bcast_task)
    torcjulia.spmd(local_computation)
    torcjulia.spmd(reduce_task)
end

torcjulia.init(main)