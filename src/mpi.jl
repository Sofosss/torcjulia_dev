"""
Minimal dummy MPI module used to run torcjulia in 
single-process mode when the "real" MPI module cannot be imported
"""

module dummyMPI
export COMM, THREAD_MULTIPLE, COMM_WORLD, Barrier, Comm_rank, Comm_size, Query_thread

struct COMM
    rank::UInt64
    num_procs::UInt64
end

struct Request
    completed::Bool
end

const THREAD_MULTIPLE = 3
const COMM_WORLD = COMM(0, 1)

Barrier(::COMM) = nothing
Ibarrier(::COMM) = Request(true)
Test(req::Request) = req.completed
Comm_rank(c::COMM) = c.rank
Comm_size(c::COMM) = c.num_procs
Query_thread() = THREAD_MULTIPLE
Finalize() = nothing

end