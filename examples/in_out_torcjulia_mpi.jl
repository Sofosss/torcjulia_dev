"""
MPI Integration Example

Demonstrates torcjulia's compatibility with direct MPI operations both inside and outside 
the runtime environment. Users can seamlessly mix torcjulia task-based parallelism with 
traditional MPI (collective) operations

The example shows:
- MPI operations BEFORE torcjulia initialization (MPI.gather, MPI.Allreduce)
- MPI operations INSIDE torcjulia tasks (same operations within task functions)
- multiple torcjulia.start() calls with MPI_finalize control

Key features:
- `MPI_finalize = false`: Keeps MPI alive between torcjulia sessions
- direct MPI calls coexist with task-based execution

Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 2 in_out_torcjulia_mpi.jl
"""

using torcjulia

using MPI


_size = MPI.Comm_size(MPI.COMM_WORLD)
_rank = MPI.Comm_rank(MPI.COMM_WORLD)   

function mpi_gather_outside()
    local_val = [_rank + 1]
    all_vals = MPI.gather(local_val, MPI.COMM_WORLD)
    
    if _rank == 0
        println("outside torcjulia (gather) -> $(vcat(all_vals...))")
    end
end

mpi_gather_outside()

function mpi_gather_inside_task()
    user_comm = MPI.COMM_WORLD
    local_val = [_rank + 1]
    all_vals = MPI.gather(local_val, user_comm)

    if _rank == 0
        println("inside torcjulia (gather) -> $(vcat(all_vals...))")
    end
end

function torc_gather()
    _ = [torcjulia.submit(mpi_gather_inside_task, qid = i) for i in 0:_size - 1]
    torcjulia.wait()
end

torcjulia.init(torc_gather; MPI_finalize = false)

function mpi_allreduce_outside()
    local_val = _rank + 1
    total_sum = MPI.Allreduce(local_val, +, MPI.COMM_WORLD)
    
    if _rank == 0
        println("outside torcjulia (Allreduce) -> $total_sum")
    end
end

mpi_allreduce_outside()

function mpi_allreduce_inside_task()
    user_comm = MPI.COMM_WORLD
    local_val = _rank + 1
    total_sum = MPI.Allreduce(local_val, +, user_comm)
    
    if _rank == 0
        println("inside torcjulia (Allreduce) -> $total_sum")
    end
end

function torc_allreduce()
    _ = [torcjulia.submit(mpi_allreduce_inside_task, qid = i) for i in 0:_size - 1]
    torcjulia.wait()
end

torcjulia.init(torc_allreduce)