module torcjulia

include("runtime.jl")

import .runtime: submit, result, input, kw_input, wait, finalize, init,
                 launch, spmd, map, starmap, reduce, map_async, mapReduce, 
                 ready, get, filter, starmap_async, scan, shm_alloc, mapPairs, 
                 mapPairsReduce, ctx, _torc_tls_get_id, gettime, set_logger_level, 
                 _enable_stealing, TaskPriority, TorcTask, TORC_NUM_WORKERS

export submit, result, input, kw_input, wait, start, set_logger_level, spmd, map, starmap, scan, shm_alloc, reduce, 
       ready, get, map_async, starmap_async, mapReduce, filter, mapPairs, mapPairsReduce, TaskPriority, TorcTask, gettime

const node_id = () -> runtime.ctx[].rank
const num_nodes = () -> runtime.ctx[].num_procs
const num_local_workers = () -> runtime.TORC_NUM_WORKERS
const num_workers = () -> runtime.ctx[].num_procs * runtime.TORC_NUM_WORKERS
const worker_local_id = () -> runtime._torc_tls_get_id()
const worker_id = () -> runtime.ctx[].rank * runtime.TORC_NUM_WORKERS + runtime._torc_tls_get_id()

enable_stealing() = (runtime._enable_stealing())
disable_stealing() = (runtime._disable_stealing())
set_scheduling_policy(policy::Symbol, weights::Union{Vector{Int}, Nothing}) = (runtime._set_scheduling_policy(policy, weights))

"""
    start(func::Function; MPI_finalize::Bool = true)

Initialize and launch the torcjulia runtime

1. init(): Sets up runtime infrastructure and spawns a dedicated server thread for inter-process communication (if running with multiple MPI processes)
2. launch(func):  Executes the user function on rank 0's main thread while worker threads across all MPI processes handle submitted tasks
3. finalize(MPI_finalize): Terminates server threads, gathers execution statistics and optionally finalizes MPI
"""
function start(func::Function; MPI_finalize::Bool = true)
    init()  
    launch(func) 
    finalize(MPI_finalize)        
end

end