module runtime

include("config.jl")    
using .config

include("mpi.jl")
import .dummyMPI: COMM, THREAD_MULTIPLE, COMM_WORLD, Barrier, Comm_rank, Comm_size, Query_thread

include("logger.jl")    
using Logging
import .logger:init_logger, set_logger_level

import Base: get

using ThreadPinning
using ConcurrentCollections
using DataStructures

using Match
using Printf
using Crayons

const HAS_MPI = try
    @eval using MPI
    true
catch
    false
end

logger.init_logger(Logging.Debug)

mutable struct MPIContext{T}
    comm::T
    rank::UInt
    num_procs::UInt
end

const ctx = Ref{MPIContext}()

if HAS_MPI
    @debug "MPI module was successfully imported"
    MPI.Init_thread(MPI.THREAD_MULTIPLE)
    TORC_COMM = MPI.Comm_dup(MPI.COMM_WORLD)
    ctx[] = MPIContext(TORC_COMM,
                       UInt(MPI.Comm_rank(TORC_COMM)),
                       UInt(MPI.Comm_size(TORC_COMM)))
else
    # if MPI is not available, we fall back to a dummy (custom) MPI implementation.
    # this dummy provides the same interface (rank, size, barrier, etc.) but does not perform any inter-process communication 
    # it just allows the code to run in a single-node/process environment
    @warn "MPI module could not be imported, loading dummy MPI module."
    ctx[] = MPIContext(dummyMPI.COMM_WORLD,
                       dummyMPI.Comm_rank(dummyMPI.COMM_WORLD),
                       dummyMPI.Comm_size(dummyMPI.COMM_WORLD))
end

const MPI_mod = HAS_MPI ? MPI : dummyMPI

const TORC_STEALING_ENABLED = Ref(get(ENV, "TORC_STEALING", "False") == "True")
const TORC_SERVER_YIELDTIME = (arg = tryparse(Float64, get(ENV, "TORC_SERVER_YIELDTIME", "0.01"))) === nothing ? 0.01 : arg
const TORC_WORKER_YIELDTIME = (arg = tryparse(Float64, get(ENV, "TORC_WORKER_YIELDTIME", "0.01"))) === nothing ? 0.01 : arg

@enum DistributionPolicy ROUND_ROBIN LOCAL RANDOM WRR

function _parse_policy(policy::Union{String, Symbol})
    _policy = policy isa String ? uppercase(strip(policy)) : string(policy)
    
    for _cand_policy in instances(DistributionPolicy)
        if string(_cand_policy) == _policy
            return _cand_policy
        end
    end
    
    if ctx[].rank == 0
        _supported = join([join(uppercasefirst.(split(lowercase(replace(string(inst), "_" => "-")), "-")), "-") for inst in instances(DistributionPolicy)], ", ")
        @warn "Invalid task distribution policy: $policy -> falling back to Round-Robin (supported policies: $_supported)"
    end

    DistributionPolicy(0)
end

const TORC_DISTRIBUTION_POLICY = Ref(_parse_policy(get(ENV, "TORC_DISTRIBUTION_POLICY", "ROUND_ROBIN")))

const node_weights = Ref(Int[])
const node_weights_cumsum = Ref(Int[])

const TORC_NUM_WORKERS, TORC_USE_SERVER = let
    nthreads = Threads.nthreads()
    multiprocess = (ctx[].num_procs > 1)

    # When running with more than one MPI process, torcjulia requires
    # at least 2 threads per process: 1 server thread and 1 worker thread
    if (multiprocess && nthreads < 2) && ctx[].rank == 0
        @error "torcjulia requires at least 2 threads per process when running with multiple MPI processes"
        
        MPI_mod.Abort(ctx[].comm, 1)
    end

    num_workers = multiprocess ? (nthreads - 1) : nthreads

    (num_workers, multiprocess)
end

const torc_tls_id = fill(Int(-1), Threads.nthreads()) # maps the os-level thread IDs to the torc-assigned IDs
const torc_tls_curr_task = [Ref{Dict{Symbol, Any}}() for _ in 1:TORC_NUM_WORKERS] # stores a reference to the task currently running on each worker-thread

const torc_shm = Dict{UInt, Tuple{Any, MPI_mod.Win}}() # SHM registry: maps shm ptr → (shared array, MPI window)

# thread-safe task queue for low-priority tasks (TaskPriority.low) with TORC_QUEUE_LEVELS nested levels
const torc_queue = [ConcurrentQueue{Union{Dict{Symbol, Any}, Nothing}}() for _ in 1:(TORC_QUEUE_LEVELS)] 

const torc_executed = Threads.Atomic{UInt}(0) 
const torc_created = Threads.Atomic{UInt}(0)
const torc_stole = Threads.Atomic{UInt}(0)
const torc_steal_attempts = Threads.Atomic{UInt}(0)
const torc_last_qid = Threads.Atomic{UInt}(0)
const torc_max_queue_depth = Threads.Atomic{UInt}(0)

const torc_stolen = Ref{UInt}(0) 

# event and not condition to ensure that if notify is called before wait, the waiting task won't miss the notification
# autoreset -> true, ensures that at most one task is released per notify,
# which is necessary because the event will be waited on twice (initialization && finalization)
const server_thread_lock = Threads.Event(true) 

const completed_tasks_lock = Threads.SpinLock()
const dependent_tasks_lock = Threads.SpinLock()

"""
    _torc_tls_get_id()

Return the current thread's worker ID
"""
@inline function _torc_tls_get_id()
    torc_tls_id[Threads.threadid()]
end

"""
    _torc_tls_set_id(id::Int)

Set the current thread's worker ID
"""
@inline function _torc_tls_set_id(id::Int)
    torc_tls_id[Threads.threadid()] = id
end

"""
    _torc_tls_get_curr_task()

Return a reference to the task currently running on this thread
"""
@inline function _torc_tls_get_curr_task()
    torc_tls_curr_task[Threads.threadid()][]
end

"""
    _torc_tls_set_curr_task(task::Dict{Symbol, Any})

Set the task currently running on this thread
"""
@inline function _torc_tls_set_curr_task(task::Dict{Symbol, Any})
    torc_tls_curr_task[Threads.threadid()][] = task
end

"""
    ConcurrentPriorityQueue{T}

Thread-safe priority queue of type `T`
"""
mutable struct ConcurrentPriorityQueue{T}
    queue::PriorityQueue{T, UInt}
    lock::Threads.SpinLock
    ConcurrentPriorityQueue{T}() where T = new(PriorityQueue{T, UInt}(Base.Order.Reverse), Threads.SpinLock())
end

"""
    Base.push!(pq::ConcurrentPriorityQueue, task, priority)

Add a task with `priority` to the priority queue
"""
function Base.push!(priority_queue::ConcurrentPriorityQueue{T}, item::T, priority::UInt) where T
    lock(priority_queue.lock)
    try
        enqueue!(priority_queue.queue, item, priority)
    finally
        unlock(priority_queue.lock)
    end
end

"""
    _maybepopfirst!(pq::ConcurrentPriorityQueue)

Remove and return the highest-priority task from the priority queue
"""
function _maybepopfirst!(priority_queue::ConcurrentPriorityQueue{T}) where T
    lock(priority_queue.lock)
    try
        return isempty(priority_queue.queue) ? nothing : Some(dequeue!(priority_queue.queue))
    finally
        unlock(priority_queue.lock)
    end
end

"""
    Base.isempty(pq::ConcurrentPriorityQueue)

Check whether the priority queue is empty
"""
Base.isempty(priority_queue::ConcurrentPriorityQueue) = begin
    lock(priority_queue.lock)
    try
        return isempty(priority_queue.queue)
    finally
        unlock(priority_queue.lock)
    end
end

const torc_priority_queue = ConcurrentPriorityQueue{Dict{Symbol, Any}}()

"""
    TorcTask

A task holding a dict with all task info
"""
# not type stable. maybe we should fall back to raw struct fields
mutable struct TorcTask
    desc::Dict{Symbol,Any}   
end

"""
    input(task::TorcTask)

Return the positional arguments of the task
"""
@inline input(task::TorcTask) = task.desc[:args]

"""
    kw_input(task::TorcTask)

Return the keyword arguments of the task
"""
@inline kw_input(task::TorcTask) = task.desc[:kwargs]

"""
    result(task::TorcTask)

Return the result-output of the executed task
"""
@inline result(task::TorcTask) = task.desc[:out]

"""
    Base.finalize(task::TorcTask)

Clear references held by the task before it is GCollected
"""
function Base.finalize(task::TorcTask)
    empty!(task.desc) 
end

@enum TaskState pending running completed

struct TaskPriorityLevel
    level::Symbol
    val::UInt
end

const TaskPriority = (
    low = TaskPriorityLevel(:low, UInt(1)),
    medium = TaskPriorityLevel(:medium, UInt(2)),
    high = TaskPriorityLevel(:high, UInt(3))
)

# thread-safe queues storing runtime info for torcjulia nodes
const node_created_tasks = ConcurrentQueue{TorcTask}()
const node_tasks_pending_time = ConcurrentQueue{UInt}()
const node_tasks_execution_time = ConcurrentQueue{UInt}()

"""
    init()

Initialize the runtime for torcjulia (starts the server thread if MPI num_procs > 1)
"""
function init()
    if ctx[].rank == 0

        provided = MPI_mod.Query_thread()
        
        @warn "MPI.Query_thread returns $(parse(Int, match(r"\d+", string(provided)).match))"

        if provided < MPI_mod.THREAD_MULTIPLE
            @warn "Warning: MPI.Query_thread returns $provided < $(MPI_mod.THREAD_MULTIPLE)"
        else
            @warn "Info: MPI.Query_thread returns MPI.THREAD_MULTIPLE"
        end

        # initialize the task info for the main thread of the master node. this task corresponds to the init func provided by the user to start torcjulia
        main_task = Dict{Symbol, Any}(
            :deps   => Threads.Atomic{UInt}(0),
            :parent => UInt(0),
            :level  => UInt(0),
            :completed => Vector{Dict{Symbol, Any}}(),
            :state => running
        )   

        main_task[:mytask] = UInt(pointer_from_objref(main_task))
        _torc_tls_set_id(1); _torc_tls_set_curr_task(main_task)

    end

    if TORC_USE_SERVER
        # start the server tpinned to the last thread of the thread pool for each MPI process
        ThreadPinning.@spawnat Threads.nthreads() server_thread()
        Base.wait(server_thread_lock)

    end

    Threads.atomic_xchg!(torc_last_qid, ctx[].rank * TORC_NUM_WORKERS)
end

"""
    launch(func::Function)

Run the user-provided function on the main thread of the root process and start worker threads to handle 
upcoming submitted tasks (ensuring proper initialization and finalization)
"""
function launch(func::Function)
    roles = Vector{Function}(undef, TORC_NUM_WORKERS)
    roles[1] = () -> (ctx[].rank == 0 ? (func(); _finalize()) : worker(1))
    for i in 2:TORC_NUM_WORKERS
        roles[i] = () -> (worker(i))
    end

    ctx[].rank != 0 || println(Crayon(foreground = :magenta)("torcjulia: main starts"))

    MPI_mod.Barrier(ctx[].comm)
    
    Threads.@threads :static for i in 1:TORC_NUM_WORKERS
        roles[i]()
    end

    _remote_workers_sync()
    
    ctx[].rank == 0 && (task = _torc_tls_get_curr_task(); task[:state] = completed; _terminate_server_threads())
end

"""
    submit(func::Function, args...; qid::Int = -1, callback::Union{Function, Nothing} = nothing,
           async_callback::Bool = true, counted::Bool = true, forced_node::Bool = false,
           priority::TaskPriorityLevel = TaskPriority.low, kwargs...)

Create and submit a TorcTask for execution

`args` and `kwargs` are passed to the user function  
`qid` selects the MPI node where the task will be sent (default: round-robin) 
`callback` is called on task completion; `async_callback` controls whether it's executed asynchronously 
`counted` tracks the task in statistics
`forced_node` indicates whether the task is pinned to the node defined by `qid`; with task stealing enabled, it may execute on a different node
`priority` sets the task priority
"""
function submit(func::Function, args...; qid::Int = -1, callback::Union{Function, Nothing} = nothing, async_callback::Bool = true, counted::Bool = true, forced_node::Bool = false, 
                priority::TaskPriorityLevel = TaskPriority.low, distribution_policy::Union{Function, Symbol, Nothing} = nothing, kwargs...)
    
    # qid (when given by the user) corresponds to the rank of the MPI process that holds the desired queue
    if qid != -1 && (qid < 0 || qid >= ctx[].num_procs)
        @error "submit: invalid qid value ($qid) - qid must be between 0 and $(ctx[].num_procs - 1)"
        throw(ArgumentError("submit: invalid qid value ($qid)"))
    end

    qid = !isnothing(distribution_policy) ? _assign_task(distribution_policy) : qid
    
    # if SHM arrays exist in args/kwargs, replace them with shm ptr
    _args, _kwargs = !isempty(torc_shm) ? (_parse_sharedmem_args(args...; kwargs...)) : (args, NamedTuple(kwargs))

    task = create_task(func, callback, async_callback, counted, forced_node, priority, _args...; _kwargs...)

    task[:mytask] = UInt(pointer_from_objref(task))

    Threads.atomic_max!(torc_max_queue_depth, task[:level])

    # use distribution policy to choose node if unspecified
    if qid == -1
       qid = _assign_task(TORC_DISTRIBUTION_POLICY[]); task[:forced_node] = false
    end
    
    task[:cbtask] = !isnothing(callback) ? create_cb_task(callback, counted, priority, task[:level]) : nothing

    counted ? Threads.atomic_add!(torc_created, UInt(1 + (!isnothing(callback) ? 1 : 0))) : nothing

    task_f = TorcTask(task); task[:myfuture] = UInt(pointer_from_objref(task_f))
    push!(node_created_tasks, task_f)

    # identify task dependencies in args/kwargs to enforce execution order
    _has_dependencies = _capture_args_deps(task)

    if !_has_dependencies
        ctx[].rank == qid ? enqueue(task[:level], task) : begin
            task[:type] = "enqueue"
            MPI_mod.send(task, ctx[].comm; dest = qid, tag = TORC_SERVER_TAG)
        end
    else
        task[:qid] = qid
    end

    task_f
end

"""
    create_task(f::Function, callback::Union{Function, Nothing}, async_callback::Bool,
                counted::Bool, forced_node::Bool, priority::TaskPriorityLevel,
                args...; kwargs...)

Create a dict containing all info needed to execute the submitted task
"""
function create_task(f::Function, callback::Union{Function, Nothing}, async_callback::Bool, counted::Bool, forced_node::Bool, 
                     priority::TaskPriorityLevel, args...; kwargs...)
    parent = _torc_tls_get_curr_task()
    Threads.atomic_add!(parent[:deps], UInt(1)) 

    Dict(
        :f => f,
        :callback => callback,
        :async_callback => async_callback,
        :counted => counted,
        :deps => Threads.Atomic{UInt}(0),
        :completed => Vector{Dict{Symbol, Any}}(),
        :dependent => Vector{Dict{Symbol, Any}}(),
        :homenode => ctx[].rank,
        :parent => UInt(pointer_from_objref(parent))  ,
        :level => min(parent[:level] + 1, TORC_QUEUE_LEVELS),
        :state => pending,
        :forced_node => forced_node ? true : false,
        :priority => priority,
        :submit_t => time_ns(),
        :out => nothing,
        :args => args,
        :kwargs => kwargs,
    )
end

"""
    create_cb_task(f, counted, priority, level)

Create a dict describing a callback task
"""
function create_cb_task(f::Function, counted::Bool, priority::TaskPriorityLevel, level::UInt)
    parent = _torc_tls_get_curr_task()
    Threads.atomic_add!(parent[:deps], UInt(1)) 

    Dict(
        :f => f,
        :callback => nothing,
        :counted => counted,
        :deps => Threads.Atomic{UInt}(0),
        :parent => UInt(pointer_from_objref(parent)),
        :completed => Vector{Dict{Symbol, Any}}(),
        :dependent => Vector{Dict{Symbol, Any}}(),
        :homenode => ctx[].rank,
        :level => level,
        :state => pending,
        :forced_node => true,
        :priority => priority,
        :out => nothing,
        :args => (),
        :kwargs => NamedTuple()
    )   
end

"""
    worker(torc_id::Int)

Worker loop that fetches tasks from the local queue and executes them.  
If the local queue is empty and work stealing is enabled, attempts to steal a task  
from another node. Continues until all user-submitted tasks have completed
"""
function worker(torc_id::Int)
    _torc_tls_set_id(torc_id)
    while true
        task = dequeue()
        isnothing(task) ? break : nothing

        if isempty(task)
            if ctx[].num_procs > 1 && TORC_STEALING_ENABLED[]
                task = steal()
                Threads.atomic_add!(torc_steal_attempts, UInt(1))
                if task[:type] == "nowork"
                    sleep(TORC_WORKER_YIELDTIME)
                    continue
                else
                    task[:counted] ? Threads.atomic_add!(torc_stole, UInt(1)) : nothing
                end
            else
                sleep(TORC_WORKER_YIELDTIME)
                continue
            end
        end

        ctx[].num_procs > 1 ? sleep(TORC_TASK_YIELDTIME) : nothing
        task[:start_t] = time_ns()
        work(task)
    end
end

"""
    work(task::Dict{Symbol, Any})

Execute the given task and handle its completion
"""

function work(task::Dict{Symbol, Any})
    _torc_tls_set_curr_task(task)

    task[:state] = running; func = task[:f]; args = task[:args]; kwargs = task[:kwargs]
    
    # if SHM arrays exist in args/kwargs, restore them from the shm ptr
    _args, _kwargs = !isempty(torc_shm) ? (_restore_sharedmem_args(args...; kwargs...)) : (args, NamedTuple(kwargs))

    y = func(_args...;  _kwargs...) # unpacking of args into individual args

    task[:out] = y
    task[:counted] ? Threads.atomic_add!(torc_executed, UInt(1)) : nothing
    task[:end_t] = time_ns()

    handle_completed_task(task)
end

"""
    handle_completed_task(task::Dict{Symbol, Any})

Finalize a completed task by updating its state, 
processing any associated callbacks and delivering results to dependent tasks || back to the node that created the task
"""
function handle_completed_task(task::Dict{Symbol, Any})
    if ctx[].rank == task[:homenode]
        is_stolen = (get(task, :type, nothing) == "stolen")

        if is_stolen
            native_task = unsafe_pointer_to_objref(Ptr{UInt}(task[:mytask]))
            native_task[:out] = task[:out]
            native_task[:start_t] = task[:start_t]; native_task[:end_t] = task[:end_t]
            parent = unsafe_pointer_to_objref(Ptr{UInt}(native_task[:parent]))
            Threads.atomic_sub!(parent[:deps], UInt(1))  
            lock(completed_tasks_lock); push!(parent[:completed], native_task); unlock(completed_tasks_lock)

            _capture_time_stats(native_task[:submit_t], native_task[:start_t], native_task[:end_t])
            
            !isnothing(native_task[:callback]) ? exec_task_callback(native_task) : nothing

            native_task[:state] = completed

            for task in native_task[:dependent]
                Threads.atomic_sub!(task[:deps], UInt(1))

                if task[:deps][] == 0
                    ctx[].rank == task[:qid] ? enqueue(task[:level], task) : begin
                                task[:type] = "enqueue"; MPI_mod.send(task, ctx[].comm; dest = task[:qid], tag = TORC_SERVER_TAG)
                            end
                end
            end
        
        else
            parent = unsafe_pointer_to_objref(Ptr{UInt}(task[:parent])) 
            Threads.atomic_sub!(parent[:deps], UInt(1))
            lock(completed_tasks_lock); push!(parent[:completed], task); unlock(completed_tasks_lock)

            _capture_time_stats(task[:submit_t], task[:start_t], task[:end_t])
            
            !isnothing(task[:callback]) ? exec_task_callback(task) : nothing

            lock(completed_tasks_lock); task[:state] = completed; unlock(completed_tasks_lock)

            for task in task[:dependent]
                Threads.atomic_sub!(task[:deps], UInt(1))

                if task[:deps][] == 0
                    ctx[].rank == task[:qid] ? enqueue(task[:level], task) : begin
                                task[:type] = "enqueue"; MPI_mod.send(task, ctx[].comm; dest = task[:qid], tag = TORC_SERVER_TAG)
                            end
                end
            end
        end

    else
        task[:type] = "answer"; dest = task[:homenode]
        # keep only the essential fields for the server thread on the homenode
        # type → label identifying the kind of message (i.e., answer)
        # mytask → pointer to the original task dictionary (on the homenode)
        # out → the result of the completed task
        Base.filter!(((k,_),) -> k in (:type, :mytask, :out, :start_t, :end_t), task)
        
        MPI_mod.send(task, ctx[].comm; dest = dest, tag = TORC_SERVER_TAG)
    end
end

"""
    steal()

Attempt to steal a task from the queue of other MPI nodes
"""
function steal()
    task = Dict(:type => "nowork"); request = Dict(:type => "steal")
    for node in Base.filter(x -> x != ctx[].rank, 0:ctx[].num_procs - 1)
        torc_terminate_flag && return task

        MPI_mod.send(request, ctx[].comm; dest = node, tag = TORC_SERVER_TAG)

        flag = false
        while !flag
            flag = MPI_mod.Iprobe(ctx[].comm; source = node, tag = TORC_STEAL_RESPONSE_TAG)
            flag || sleep(TORC_WORKER_YIELDTIME)
        end

        response = MPI_mod.recv(ctx[].comm; source = node, tag = TORC_STEAL_RESPONSE_TAG)

        response[:type] == "nowork" || return response
    end
    
    task
end

"""
    server_thread()

Continuously process incoming MPI messages, manage tasks and callbacks, and handle node termination including its worker threads
"""
function server_thread()
    _torc_tls_set_id(Threads.nthreads())
    global torc_terminate_flag = false
    
    notify(server_thread_lock)
    
    while true
        flag, status = false, nothing
        while !flag
            flag, status = MPI_mod.Iprobe(ctx[].comm, MPI_mod.Status; source = MPI_mod.ANY_SOURCE, tag = TORC_SERVER_TAG)
            flag || sleep(TORC_SERVER_YIELDTIME)
        end
        
        task = MPI_mod.recv(ctx[].comm; source = status.source, tag = TORC_SERVER_TAG)

        @match task[:type] begin 
            "exit" => begin
                # notify the master node's server thread to terminate
                ctx[].rank == 1 && _notify_master_server_thread()
                break

            end

            "terminate" => begin
                global torc_terminate_flag = true
                # add TORC_NUM_WORKERS termination signals to the first level so that all worker threads are notified to terminate
                foreach(_ -> enqueue(UInt(1), nothing), 1:TORC_NUM_WORKERS)
                
            end

            "enqueue" => begin
                enqueue(task[:level], task)

            end

            "answer" => begin
                native_task = unsafe_pointer_to_objref(Ptr{UInt}(task[:mytask]))
                native_task[:out] = task[:out]; native_task[:state] = completed
                native_task[:start_t] = task[:start_t]; native_task[:end_t] = task[:end_t]
                parent = unsafe_pointer_to_objref(Ptr{UInt}(native_task[:parent]))
                Threads.atomic_sub!(parent[:deps], UInt(1))  
                lock(completed_tasks_lock); push!(parent[:completed], native_task); unlock(completed_tasks_lock)

                _capture_time_stats(native_task[:submit_t], native_task[:start_t], native_task[:end_t])

                !isnothing(native_task[:callback]) ? exec_task_callback(native_task) : nothing

                for task in native_task[:dependent]
                    Threads.atomic_sub!(task[:deps], UInt(1))
                    
                    if task[:deps][] == 0
                        ctx[].rank == task[:qid] ? enqueue(task[:level], task) : begin
                                    task[:type] = "enqueue"; MPI_mod.send(task, ctx[].comm; dest = task[:qid], tag = TORC_SERVER_TAG)
                                end
                    end
                end

            end

            "steal" => begin
                task = dequeue_steal()
                t = isnothing(task) ? (enqueue(UInt(1), nothing); Dict(:type => "nowork")) :
                begin
                    task[:type] = isempty(task) ? "nowork" : "stolen"
                    if task[:type] == "stolen" && task[:counted]
                        torc_stolen[] += 1
                    end
                    task
                end

                MPI_mod.send(t, ctx[].comm; dest = status.source, tag = TORC_STEAL_RESPONSE_TAG)
                
            end
            
            _ => begin
                @warn "Unknown task type: $(task[:type])"
            end
        end
    end

    notify(server_thread_lock)
end

"""
    enqueue(level::UInt, task)

Add a task to the appropriate queue based on its priority (default priority: low) and (parallelism) level
"""
@inline function enqueue(level::UInt, task::Union{Dict{Symbol, Any}, Nothing})
    if isnothing(task) || (haskey(task, :priority) && task[:priority] == TaskPriority.low)
        push!(torc_queue[level], task)
    else
        push!(torc_priority_queue, task, task[:priority].val)
    end
    
    nothing
end

"""
    dequeue()

Return the next available task, prioritizing high-priority tasks first
"""
function dequeue()
    # check priority tasks first
    task = _maybepopfirst!(torc_priority_queue)
    if !isnothing(task)
        return something(task) 
    end

    # try to get a task from innermost to outermost level
    # (innermost tasks have highest priority (&& granularity -> coarse-grained) and the idea is to remain local to the node that submitted them)
    for level in 1:TORC_QUEUE_LEVELS
        task = maybepopfirst!(torc_queue[level]) # -> returns (Some(T), nothing)
        if !isnothing(task)
            return something(task) 
        end
    end

    Dict{Symbol, Any}()
    # phadjido had sleep() in both functions, which caused x2 idle time - we keep sleep() only in the _task_scheduler_loop()
end

"""
    dequeue_steal()

Return the next available task that can be stolen, skipping tasks pinned to a specific node
"""
function dequeue_steal()
    # check priority tasks first
    task_opt = _maybepopfirst!(torc_priority_queue)
    if !isnothing(task_opt)
        task = something(task_opt)
        # tasks with forced_node cannot be stolen
        if !task[:forced_node]
            return task
        
        end

        # push back because this task is pinned to a specific node and must not be stolen; reinserting prevents it from being lost
        push!(torc_priority_queue, task, task[:priority].val)
    end

    # attempt to steal tasks from higher levels first (fine-grained, outermost tasks)
    for level in TORC_QUEUE_LEVELS:-1:1
        task_opt = maybepopfirst!(torc_queue[level])
        
        if !isnothing(task_opt)
            task = something(task_opt)
            if !(task isa Dict) || !task[:forced_node]
                return task
    
            end
        
            enqueue(level, task) 
        end
    end

    Dict{Symbol, Any}()
end

"""
    wait_tasks_scheduler_loop()

Continuously fetch and execute tasks, keeping workers busy while waiting for the completion of specific tasks (_waitspec) or all tasks (_waitall)
"""
function wait_tasks_scheduler_loop()
    while true
        task = dequeue()
        isnothing(task) ? break : nothing

        if isempty(task)
            if ctx[].num_procs > 1 && TORC_STEALING_ENABLED[]
                task = steal()
                Threads.atomic_add!(torc_steal_attempts, UInt(1))
                if task[:type] == "nowork"
                    sleep(TORC_WORKER_YIELDTIME)
                    break
                end
                task[:counted] ? Threads.atomic_add!(torc_stole, UInt(1)) : nothing
            else
                sleep(TORC_WORKER_YIELDTIME)
                break
            end
        end

        ctx[].num_procs > 1 ? sleep(TORC_TASK_YIELDTIME) : nothing
        task[:start_t] = time_ns()
        work(task)
    end

    nothing
end

"""
    wait(tasks = nothing; return_completed = false)

Wait for the given tasks or all submitted tasks to complete
"""
function wait(tasks::Union{Vector{TorcTask}, Nothing} = nothing; return_completed::Bool = false)
    isnothing(tasks) || return _waitspec(tasks, return_completed)
   
    _waitall(return_completed)

end

"""
    _waitall(return_completed::Bool)

Block until all child tasks of the current task are completed
"""
function _waitall(return_completed::Bool)
    main_task = _torc_tls_get_curr_task()

   # the thread executing the task that called wait() becomes a worker thread until all its child tasks have completed
    while main_task[:deps][] > 0
        wait_tasks_scheduler_loop()
    end

    _torc_tls_set_curr_task(main_task)

    return_completed ? [unsafe_pointer_to_objref(Ptr{UInt}(task[:myfuture])) for task in main_task[:completed]] : nothing
end

"""
    _waitspec(tasks::Vector{TorcTask}, return_completed::Bool; ref_task = nothing)

Wait for the specified tasks to complete
"""
function _waitspec(tasks::Vector{TorcTask}, return_completed::Bool; ref_task::Union{UInt, Nothing} = nothing)
    main_task = isnothing(ref_task) ? _torc_tls_get_curr_task() : unsafe_pointer_to_objref(Ptr{UInt}(ref_task))
    rec_tasks = Dict{UInt, Bool}()
    pending_tasks = UInt[]

    for task in tasks
        t = task.desc[:mytask]
        if !haskey(rec_tasks, t)
            rec_tasks[t] = true; push!(pending_tasks, t)
        else
            println("Duplicate task detected: $t — will be ignored")
        end
    end

    curr_completed_tasks = [task[:mytask] for task in main_task[:completed]]

    # the thread executing the task that called wait() becomes a worker thread until all the specified (child) tasks have completed
    while !all(t -> t in curr_completed_tasks, pending_tasks)
        wait_tasks_scheduler_loop()
        curr_completed_tasks = [task[:mytask] for task in main_task[:completed]]
    end

    _torc_tls_set_curr_task(main_task)

    return_completed ? [task for task in tasks if task.desc[:mytask] in curr_completed_tasks] : nothing # condition -> avoid duplicates
end

"""
    _remote_workers_sync()

Block until all MPI nodes reach the barrier
"""

function _remote_workers_sync()
    # barrier: ensures all worker threads across all MPI processes have terminate
    req = MPI_mod.Ibarrier(ctx[].comm)
    
    flag = false
    while !flag
        flag = MPI_mod.Test(req)
        flag || sleep(TORC_WORKER_YIELDTIME)  
    end

    nothing
end

"""
    _terminate_nodes()

Send termination signals to all remote MPI nodes
"""
function _terminate_nodes() 
    foreach(node -> MPI_mod.send(Dict(:type => "terminate"), ctx[].comm; dest = node, tag = TORC_SERVER_TAG), 1:(ctx[].num_procs - 1))

    nothing
end

"""
    _notify_master_server_thread()

Send termination signal to the master node's server thread
"""

function _notify_master_server_thread()
    MPI_mod.send(Dict(:type => "exit"), ctx[].comm; dest = 0, tag = TORC_SERVER_TAG)

    nothing
end

"""
    _terminate_server_threads()

Send termination signals to all server threads on remote MPI nodes
"""
function _terminate_server_threads()
    foreach(node -> MPI_mod.send(Dict(:type => "exit"), ctx[].comm; dest = node, tag = TORC_SERVER_TAG), 1:(ctx[].num_procs - 1))

    nothing
end

"""
    finalize(MPI_finalize::Bool)

Finalize the torcjulia runtime: wait for server threads exit, free shared memory if allocated and (optionally) finalize MPI
"""
function finalize(MPI_finalize::Bool)
    TORC_USE_SERVER ? Base.wait(server_thread_lock) : nothing
   
    _torc_stats(); torc_stolen[] = 0

    _free_shm()

    MPI_mod.Barrier(ctx[].comm)
    MPI_finalize ? MPI_mod.Finalize() : nothing
end

"""
    _finalize()

Signal worker threads to terminate and if running with multiple MPI nodes, notify remote nodes
"""
function _finalize()
    global torc_terminate_flag = true
    foreach(_ -> enqueue(UInt(1), nothing), 1:TORC_NUM_WORKERS) 

    ctx[].num_procs > 1 ? _terminate_nodes() : nothing
end

"""
    _free_shm()

Release all shared memory windows and clear the shared memory registry
"""
function _free_shm()
    isempty(torc_shm) && return
    for (_, (_, win)) in torc_shm
        MPI.free(win)
    end
    empty!(torc_shm)
end

"""
    exec_task_callback(task::Dict{Symbol, Any})

Execute the callback associated with a completed task if synchronous, or enqueue it for later execution if asynchronous
"""
function exec_task_callback(task::Dict{Symbol, Any})
    cb_task = task[:cbtask]
    # cb_task[:args] = unsafe_pointer_to_objref(Ptr{UInt}(task[:myfuture]))
    cb_task[:args] = task[:out]
    cb_task[:submit_t] = time_ns()

    # if the callback is async, submit it to the queue for later execution, otherwise execute it immediately
    task[:async_callback] ? enqueue(cb_task[:level], cb_task) : begin
        cb_task[:start_t] = time_ns(); cb_task[:f](cb_task[:args]); cb_task[:end_t] = time_ns()
        parent = unsafe_pointer_to_objref(Ptr{UInt}(cb_task[:parent]))
        Threads.atomic_sub!(parent[:deps], UInt(1)) 
        task[:counted] ? Threads.atomic_add!(torc_executed, UInt(1)) : nothing
        _capture_time_stats(cb_task[:submit_t], cb_task[:start_t], cb_task[:end_t])
    end

    nothing
end

"""
    _capture_args_deps(task::Dict{Symbol, Any})

Identify unfinished tasks in positional and keyword arguments of the given task and register dependencies accordingly 
"""
function _capture_args_deps(task::Dict{Symbol, Any})
    deps = UInt(0)
    args = task[:args]; kwargs = task[:kwargs]

    _scan(x) = begin
        if x isa Number || x isa String || x isa Type || x === nothing
            return
        end
        
        if x isa TorcTask
            lock(completed_tasks_lock) do
                if x.desc[:state] != completed
                    deps += 1

                    lock(dependent_tasks_lock) do
                        push!(x.desc[:dependent], task)
                    end

                    Threads.atomic_add!(task[:deps], UInt(1))
                end
            end
        elseif x isa AbstractArray || x isa Tuple
            foreach(_scan, x)  
        elseif x isa AbstractDict
            foreach(_scan, values(x))
        elseif isstructtype(typeof(x))
            foreach(_scan, (getfield(x, f) for f in fieldnames(typeof(x))))
        end
    end

    _scan(args); _scan(values(kwargs))
    
    deps > 0
end

"""
    _parse_sharedmem_args(args...; kwargs...)

Replace shared memory arrays in positional and keyword arguments with their corresponding shm pointers
"""
function _parse_sharedmem_args(args...; kwargs...)
    function _replace(x)
        if x isa Number || x isa String || x isa Type || x === nothing
            return x
        end

        if x isa AbstractArray
            ptr = UInt(pointer_from_objref(x))
            return haskey(torc_shm, ptr) ? ptr : x
        elseif x isa Tuple
            return Base.map(_replace, x)
        elseif x isa NamedTuple
            return Base.map(_replace, x)
        elseif x isa Dict
            return Dict(k => _replace(v) for (k,v) in x)
        elseif isstructtype(typeof(x))
            fvals = Base.map(f -> _replace(getfield(x, f)), fieldnames(typeof(x)))
            return (typeof(x))(fvals...)
        else
            return x
        end
    end

    Base.map(_replace, args), Base.map(_replace, NamedTuple(kwargs))
end

"""
    _restore_sharedmem_args(args...; kwargs...)

Restore shared memory arrays in positional and keyword arguments from their corresponding shm pointers
"""
function _restore_sharedmem_args(args...; kwargs...)
    function _restore(x)
        if x isa UInt && haskey(torc_shm, x)
            return _get_shared_mem(x)

        elseif x isa Number || x isa String || x isa Type || x === nothing
            return x

        elseif x isa Tuple
            return tuple(Base.map(_restore, x)...)

        elseif x isa NamedTuple
            return NamedTuple{keys(x)}(Base.map(_restore, values(x)))

        elseif x isa Dict
            return Dict(k => _restore(v) for (k,v) in x)

        elseif isstructtype(typeof(x))
            fvals = Base.map(f -> _restore(getfield(x, f)), fieldnames(typeof(x)))
            return (typeof(x))(fvals...)

        else
            return x
        end
    end

    Base.map(_restore, args), Base.map(_restore, NamedTuple(kwargs))
end

"""
    _agg_time_stats(pending_q::ConcurrentQueue{UInt}, exec_q::ConcurrentQueue{UInt})

Aggregate total time and count of tasks from the given queues
"""
function _agg_time_stats(pending_q::ConcurrentQueue{UInt}, exec_q::ConcurrentQueue{UInt})
    agg(q)::Tuple{UInt, UInt} = begin
        _sum::UInt = 0
        _cnt::UInt = 0
        while true
            t = maybepopfirst!(q)
            isnothing(t) && break
            _sum += something(t)
            _cnt += 1
        end
        return _sum, _cnt
    end

    t1 = Threads.@spawn agg(pending_q)
    t2 = Threads.@spawn agg(exec_q)

    res1 = fetch(t1)::Tuple{UInt, UInt}
    res2 = fetch(t2)::Tuple{UInt, UInt}

    res1, res2
end

"""
    _torc_stats()

Collect and display torcjulia runtime statistics across all MPI nodes
"""
function _torc_stats()
    created = Threads.atomic_xchg!(torc_created, UInt(0))
    executed = Threads.atomic_xchg!(torc_executed, UInt(0))
    stole = Threads.atomic_xchg!(torc_stole, UInt(0))
    steal_attempts = Threads.atomic_xchg!(torc_steal_attempts, UInt(0))
    max_queue_depth = Threads.atomic_xchg!(torc_max_queue_depth, UInt(0))

    (pending_sum, pending_cnt), (exec_sum, exec_cnt) = _agg_time_stats(node_tasks_pending_time, node_tasks_execution_time)

    _local = Float64[pending_sum, pending_cnt, exec_sum, exec_cnt]
    _global = ctx[].num_procs > 1 ? MPI.Reduce(_local, +, 0, ctx[].comm) : _local

    println(Crayon(foreground = :green)(
        @sprintf("torcjulia: node[%d]: created=%d, executed=%d, stole=%d, stolen=%d, steal_attempts=%d, max_queue_depth=%d",
                    ctx[].rank, created, executed, stole, torc_stolen[], steal_attempts, max_queue_depth)
    ))

    flush(stdout)

    MPI_mod.Barrier(ctx[].comm)

    if ctx[].rank == 0
        total_pending_sum, total_pending_cnt, total_exec_sum, total_exec_cnt = _global
        avg_pending = total_pending_cnt == 0 ? 0.0 : total_pending_sum / total_pending_cnt / 1e9
        avg_exec = total_exec_cnt == 0 ? 0.0 : total_exec_sum / total_exec_cnt / 1e9

    println(Crayon(foreground = :magenta)(
        @sprintf("torcjulia [tasks]: average pending time=%.3f s, average execution time=%.3f s",
                avg_pending, avg_exec)
    ))
    end

    flush(stdout)
    
    nothing
end

"""
    _capture_time_stats(submit_t, start_t, end_t)

Record pending and execution time of the completed task
"""
function _capture_time_stats(submit_t::UInt64, start_t::UInt64, end_t::UInt64)
    push!(node_tasks_pending_time, start_t - submit_t)
    push!(node_tasks_execution_time, end_t - start_t)
    
    nothing
end

"""
    _assign_task(policy::Union{Symbol, Function, DistributionPolicy}) -> UInt

Determine the target MPI node for task assignment based on the specified distribution policy

`policy` specifies the task distribution strategy (can be a `Symbol`, custom user-provided `Function`, or `DistributionPolicy`)

# Supported Policies
- round-robin: distributes tasks evenly across nodes in a cyclic manner
- weighted round-robin (wrr): distributes tasks proportionally based on node weights (requires prior weight configuration from the user)
- local: assigns task to the current node
- random: randomly selects a target node
- user-defined function returning a node index (0 to num_procs-1)

# Notes
- if a custom function returns an invalid node index, the policy falls back to `ROUND_ROBIN`
- WRR falls back to `ROUND_ROBIN` if weights are not configured
"""
function _assign_task(policy::Union{Symbol, Function, DistributionPolicy})
    if policy isa Function
        node = policy()
        
        if node isa Integer && 0 <= node < ctx[].num_procs
            return node
        end
        
        @warn "submit: task assignment failed - provided distribution policy returned invalid value: $node (must be an integer between 0 and $(ctx[].num_procs - 1) -> task will be reassigned"
        policy = "ROUND_ROBIN"
    end
    
    policy = policy isa Symbol ? _parse_policy(policy) : policy

    if policy == ROUND_ROBIN
        _qid = Threads.atomic_add!(torc_last_qid, UInt(1)) % (ctx[].num_procs * TORC_NUM_WORKERS)
        return trunc(UInt, _qid / TORC_NUM_WORKERS)
    end

    if policy == WRR
        if isempty(node_weights[])
            @warn "Weighted Round-Robin (WRR) requires weights to be set beforehand -> falling back to Round-Robin"
            _set_scheduling_policy(:ROUND_ROBIN)
            
            _qid = Threads.atomic_add!(torc_last_qid, UInt(1)) % (ctx[].num_procs * TORC_NUM_WORKERS)
            return trunc(UInt, _qid / TORC_NUM_WORKERS)
        end

        _qid = Threads.atomic_add!(torc_last_qid, UInt(1)) % (node_weights_cumsum[][end])
        _node_idx = searchsortedfirst(node_weights_cumsum[], _qid + 1)

        return UInt(_node_idx - 1)
    end
    
    if policy == LOCAL
        return ctx[].rank
    end

    # policy == RANDOM
    rand(0:ctx[].num_procs - 1)
end

"""
    spmd(func, args...; counted = true, return_completed = true, kwargs...)

Submit the given function with its arguments to all MPI nodes and wait until all tasks complete
"""
function spmd(func, args...; priority::TaskPriorityLevel = TaskPriority.low, counted = true, return_completed = true, kwargs...)
    # compared to the Python SPMD version, keyword arguments are also supported here (see spmd.jl)
    for node in 0:Int(ctx[].num_procs) - 1
        submit(func, args...; qid = node, priority = priority, counted = counted, kwargs...)
    end

    _waitall(return_completed)
end

"""
    map(func, iterables...; chunksize = nothing, qid = -1, counted = true, forced_node = false, priority = TaskPriority.low, args = (), kwargs...)

Parallel version of `map` that applies `func` to elements from `iterables`

When multiple iterables are provided, `func` is applied to tuples of corresponding elements 

# Keyword Arguments
- `chunksize`: number of elements per task (automatic if `nothing`)  
- `qid`: target MPI node (default: round-robin)  
- `counted`: track spawned tasks in runtime statistics  
- `forced_node`: pin tasks to specified qid  
- `priority`: tasks priority  
- `args`, `kwargs`: extra positional or keyword arguments passed to `func`  

# Returns
Flattened vector of results

# Example
- `map(f, [1,2,3])` → `[f(1), f(2), f(3)]`
- `map(f, [1,2], [3,4])` → `[f(1,3), f(2,4)]`
"""
function map(func, iterables...; chunksize::Union{Int, Nothing} = nothing, qid::Int = -1, 
                  counted::Bool = true, forced_node::Bool = false, priority::TaskPriorityLevel = TaskPriority.low, 
                  args::Tuple = (), kwargs...)
    # parallel equivalent of Python's built-in map.
    # compared to the corresponding torcpy version (see test5.jl):
    # 1. supports additional positional arguments beyond the elements of the iterable 
    # 2. supports keyword arguments

    items = zip(iterables...)

    chunksize = something(chunksize, max(cld(length(iterables[1]), ctx[].num_procs * TORC_NUM_WORKERS), 1))

    _wrapper = x -> func(x..., args...; kwargs...)
    
    futures = if chunksize == 1
        [submit(_wrapper, x; qid = qid, counted = counted, forced_node = forced_node, priority = priority) for x in items]
    else
        _ch_wrapper = ch -> [_wrapper(x) for x in ch]
        [submit(_ch_wrapper, ch; qid = qid, counted = counted, forced_node = forced_node, priority = priority) 
                for ch in Iterators.partition(items, chunksize)]
    end
    
    _waitall(false)
    
    results = [result(t) for t in futures]
    flat = chunksize == 1 ? results : vcat(results...)
    reshape(flat, size(iterables[1])...)    
end

"""
    starmap(func, iterable; chunksize = nothing, qid = -1, counted = true, 
                 forced_node = false, priority = TaskPriority.low, args = (), kwargs...)

Parallel version of Python's `itertools.starmap`, applying `func` to each element of `iterable`  
by unpacking its contents as arguments to `func`

# Keyword Arguments
- `chunksize`: number of elements per task (auto-calculated if `nothing`)  
- `qid`: target MPI node (default: round-robin)  
- `counted`: track spawned tasks in runtime statistics  
- `forced_node`: pin tasks to the specified qid  
- `priority`: tasks priority  
- `args`, `kwargs`: extra positional or keyword arguments passed to `func`  

# Returns
Flattened vector of results

# Example
- `data = [(1,2), (3,4)]`: `starmap(f, data)` → `[f(1,2), f(3,4)]
"""
function starmap(func, iterable; chunksize::Union{Int, Nothing} = nothing, qid::Int = -1, 
                      counted::Bool = true, forced_node::Bool = false, priority::TaskPriorityLevel = TaskPriority.low, 
                      args::Tuple = (), kwargs...)
    # parallel equivalent of Python's built-in starmap
    # similar to map, but each element of the iterable is itself an iterable 
    # whose elements are unpacked as args to the function.
    # data = [(1,2), (3,4)]: starmap(func, data) -> [func(1,2), func(3,4)]
    # + supports extra positional and keyword arguments
    T = Base.promote_typeof(func(iterable[1]..., args...; kwargs...)) 

   chunksize = something(chunksize, max(cld(length(iterable), ctx[].num_procs * TORC_NUM_WORKERS), 1))

    _wrapper = x -> T(func(x..., args...; kwargs...))
    
   futures = if chunksize == 1
        [submit(_wrapper, x; qid = qid, counted = counted, forced_node = forced_node, priority = priority) for x in iterable]
    else
        _ch_wrapper = ch -> begin 
            out = Vector{T}(undef, length(ch))
            @inbounds for (i,x) in enumerate(ch)
                out[i] = func(x..., args...; kwargs...)
            end
            out
        end
        
        [submit(_ch_wrapper, ch; qid = qid, counted = counted, forced_node = forced_node, priority = priority) 
        for ch in Iterators.partition(iterable, chunksize)]
    end

    _waitall(false)
    
    results = [result(t) for t in futures]
    chunksize == 1 ? results : foldl(vcat, results, init = T[])
end

mutable struct AsyncResult
    tasks::Vector{TorcTask}
end

@inline function ready(ar::AsyncResult)
    all(t -> t.desc[:state] == completed, ar.tasks)
end

function wait(ar::AsyncResult)
    _waitspec(ar.tasks, false)
end

function get(ar::AsyncResult)
    _waitspec(ar.tasks, false)
    results = [result(t) for t in ar.tasks]
    all(x -> isa(x, AbstractArray), results) ? collect(Iterators.flatten(results)) : results
end

"""
    map_async(func, iterables...; chunksize = nothing, qid = -1, counted = true, forced_node = false, 
                   priority = TaskPriority.low, callback = nothing, args = (), kwargs...)

Asynchronous parallel version of `map` (similar to Python's `multiprocessing.Pool.map_async`)
Applies `func` to each element of the zipped `iterables` asynchronously

# Keyword Arguments
- `chunksize`: number of elements per task (auto-calculated if `nothing`)  
- `qid`: target MPI node (default: round-robin)  
- `counted`: track spawned tasks in runtime statistics  
- `forced_node`: pin tasks to the specified qid  
- `priority`: tasks priority  
- `callback`: optional function called with the results once all tasks complete  
- `args`, `kwargs`: extra positional or keyword arguments passed to `func`  

# Returns
An `AsyncResult` object containing the submitted tasks
Use `ready()`, `wait()` and `get()` to monitor or/and retrieve results

# Example
res = map_async((x,y) -> x+y, [1,2,3], [4,5,6])
get(res)  # waits for completion and returns [5,7,9]
"""
function map_async(func, iterables...; chunksize::Union{Int, Nothing} = nothing, qid::Int = -1, 
                        counted::Bool = true, forced_node::Bool = false, priority::TaskPriorityLevel = TaskPriority.low, 
                        callback = nothing, args::Tuple = (), kwargs...)
    items = zip(iterables...) 

    chunksize = something(chunksize, max(cld(length(iterables[1]), ctx[].num_procs * TORC_NUM_WORKERS), 1))

    _wrapper = x -> func(x..., args...; kwargs...)

    futures = if chunksize == 1
        [submit(_wrapper, x; qid = qid, counted = counted, forced_node = forced_node, priority = priority) for x in items]
    else
        _ch_wrapper = ch -> [_wrapper(x) for x in ch]
        [submit(_ch_wrapper, ch; qid = qid, counted = counted, forced_node = forced_node, priority = priority) 
                for ch in Iterators.partition(items, chunksize)]
    end

    if !isnothing(callback)
        function _exec_cb(ref_task, tasks, cb)
            _waitspec(tasks, false; ref_task = ref_task)
            res = [result(t) for t in tasks]
            res = all(x -> isa(x, AbstractArray), res) ? vcat(res...) : res
            cb(res)
        end
        ref_task = _torc_tls_get_curr_task()[:mytask]; _node = ctx[].rank
        submit(_exec_cb, ref_task, futures, callback; qid = Int(_node), forced_node = true)
    end

    AsyncResult(futures)
end

"""
    starmap_async(func, iterable; chunksize = nothing, qid = -1, counted = true, forced_node = false,
                      priority = TaskPriority.low, callback = nothing, args = (), kwargs...)

Asynchronous parallel version of Python's `itertools.starmap` combined with `map_async`
Applies `func` to each element of `iterable` by unpacking its contents as arguments, submitting all tasks asynchronously

# Keyword Arguments
- `chunksize`: number of elements per task (auto-calculated if `nothing`)  
- `qid`: target MPI node (default: round-robin)  
- `counted`: track spawned tasks in runtime statistics  
- `forced_node`: pin tasks to the specified `qid`  
- `priority`: tasks priority  
- `callback`: optional function called with the results once all tasks complete  
- `args`, `kwargs`: additional positional or keyword arguments passed to `func`  

# Returns
An `AsyncResult` object containing the submitted tasks
Use `ready()`, `wait()` and `get()` to monitor or/and retrieve results

# Example
data = [(1,2), (3,4), (5,6)]
res = starmap_async((x,y) -> x+y, data)
get(res)  # waits for completion and returns [3,7,11]
"""

function starmap_async(func, iterable; chunksize::Union{Int, Nothing} = nothing, qid::Int = -1, 
                            counted::Bool = true, forced_node::Bool = false, priority::TaskPriorityLevel = TaskPriority.low, 
                            callback = nothing, args::Tuple = (), kwargs...)
    chunksize = something(chunksize, max(cld(length(iterable), ctx[].num_procs * TORC_NUM_WORKERS), 1))

    _wrapper = x -> func(x..., args...; kwargs...)

    futures = if chunksize == 1
        [submit(_wrapper, x; qid = qid, counted = counted, forced_node = forced_node, priority = priority) for x in iterable]
    else
        _ch_wrapper = ch -> [_wrapper(x) for x in ch]
       [submit(_ch_wrapper, ch; qid = qid, counted = counted, forced_node = forced_node, priority = priority) 
                for ch in Iterators.partition(iterable, chunksize)]
    end

    if !isnothing(callback)
        function _exec_cb(ref_task, tasks, cb)
            _waitspec(tasks, false; ref_task = ref_task)
            res = [result(t) for t in tasks]
            res = all(x -> isa(x, AbstractArray), res) ? vcat(res...) : res
            cb(res)
        end
        ref_task = _torc_tls_get_curr_task()[:mytask]; _node = ctx[].rank
        submit(_exec_cb, ref_task, futures, callback; qid = Int(_node), forced_node = true)
    end

    AsyncResult(futures)
end

"""
    reduce(func, data; chunksize = nothing, direction = :column, mode = "1d", qid = -1,
                counted = true, forced_node = false, priority = TaskPriority.low, args = (), kwargs...)

Parallel associative reduction over a vector or matrix
Supports 1D or 2D reductions (optional row-wise or column-wise operations when reducing a matrix)

# Arguments
- `func`: associative function used to reduce elements (e.g., `+`, `*`, `max`)  
- `data`: vector or matrix to reduce  

# Keyword Arguments
- `chunksize`: number of elements per task (auto-calculated if `nothing`)  
- `direction`: `:column` or `:row` (only relevant for 2D reductions)  
- `mode`: `"1d"` or `"2d"`; `"2d"` reduces across rows/columns to produce a scalar  
- `qid`: target MPI node (default: round-robin)  
- `counted`: track spawned tasks in runtime statistics  
- `forced_node`: pin tasks to the specified `qid`  
- `priority`: tasks priority  
- `args`, `kwargs`: additional positional or keyword arguments passed to `func`  

# Returns
Reduced value: a scalar for 2D reductions, or a vector/scalar for 1D reductions depending on input

# Example
data = [1, 2, 3, 4]
reduce(+, data)  # returns 10

mat = [1 2; 3 4]
reduce(+, mat, direction = :column, mode = "2d")  # returns 10
reduce(+, mat, direction = :column, mode = "1d")  # returns [4,6] (column-wise sum)
"""
function reduce(func, data; chunksize::Union{Int, Nothing} = nothing, direction::Symbol = :column, mode::String = "1d", qid::Int = -1, 
                     counted::Bool = true, forced_node::Bool = false, priority::TaskPriorityLevel = TaskPriority.low, 
                     args::Tuple = (), kwargs...)
    # performs associative reductions. supports 1D/2D (2D: supported only for matrices) reductions on vectors or matrices (row-wise or column-wise)
    # (producing scalars or vectors)

    if !(direction in (:column, :row))
        @error "direction must be :column or :row"
        MPI_mod.Abort(ctx[].comm, 1)
    end

    if !(mode in ("1d", "2d"))
        @error "mode must be \"1d\" or \"2d\""
        MPI_mod.Abort(ctx[].comm, 1)
    end

    if mode == "2d" && isa(data, AbstractVector) 
        @error "2D reduction requires a matrix."
        MPI_mod.Abort(ctx[].comm, 1)
    end

    isa(chunksize, Int) && chunksize == 1 && (@warn "Reduction chunksize cannot be 1. Setting to 2."; chunksize = 2)
    
    ndim = isa(data, AbstractVector) ? length(data) : (direction === :column ? size(data, 1) : size(data, 2))
    if ndim == 1
        return (data isa AbstractVector ? data[1] : (mode === "1d" ? (direction === :column ? data[1, :] : data[:, 1]) :
                (direction === :column ? reduce(func, data[1, :]; chunksize = chunksize, qid = qid, 
                forced_node = forced_node, priority = priority, args = args, kwargs...) 
                : reduce(func, data[:, 1]; chunksize = chunksize, qid = qid, 
                forced_node = forced_node, priority = priority, args = args, kwargs...))))
    end
    
    chunksize = something(chunksize, max(cld(ndim, ctx[].num_procs * TORC_NUM_WORKERS), 2))

    local_reduce(x) = begin
        if isa(x, AbstractVector)
            acc = x[1]
            @inbounds for i in 2:length(x)
                acc = func(acc, x[i], args...; kwargs...)
            end
            return acc
        else
            rows, cols = size(x)
            if direction === :column
                acc = [x[1, j] for j in 1:cols]
                @inbounds for j in 1:cols
                    @simd for i in 2:rows
                        acc[j] = func(acc[j], x[i,j], args...; kwargs...)
                    end
                end
            else
                acc = [x[i, 1] for i in 1:rows]
                @inbounds for i in 1:rows
                    @simd for j in 2:cols
                        acc[i] = func(acc[i], x[i,j], args...; kwargs...)
                    end
                end
            end
            return acc
        end
    end

    current = data
    while true
        len = isa(current, AbstractVector) ? length(current) :
              (direction === :column ? size(current, 1) : size(current, 2))

        if len ≤ chunksize
            res = len == 1 ? (current isa AbstractVector ? current[1] :
                               direction === :column ? current[1, :] : current[:, 1]) :
                  local_reduce(current)
            break
        end

        if current isa AbstractVector
            @views chunks = [current[i:min(i+chunksize-1, len)] for i in 1:chunksize:len]
        elseif direction === :column
            @views chunks = [current[i:min(i+chunksize-1, len), :] for i in 1:chunksize:len]
        else
            @views chunks = [current[:, i:min(i+chunksize-1, len)] for i in 1:chunksize:len]
        end

        futures = submit.(local_reduce, chunks; qid = qid, counted = counted, forced_node = forced_node, priority = priority)
        _waitall(false)
        partials = result.(futures)

        if all(p -> p isa AbstractVector, partials)
            current = direction === :column ? vcat(reshape.(partials, 1, :)...) :
                                             hcat(reshape.(partials, :, 1)...)
        else
            current = partials  
        end
    end

    # second step of 2d reduction > from vector to scalar
    if mode == "2d"
        while length(res) > chunksize
            chunks = [@view res[i:min(i+chunksize-1, length(res))] for i in 1:chunksize:length(res)]
            futures = submit.(local_reduce, chunks; qid = qid, counted = counted, forced_node = forced_node, priority = priority)
            _waitall(false)
            res = result.(futures)
        end
        res = res isa AbstractVector ? (length(res) == 1 ? res[1] : local_reduce(res)) : res
    end

    res
end

"""
    mapReduce(map_func, reduce_func, iterables...; chunksize = nothing, direction = :column, mode = "1d", qid = -1,
                   counted = true, forced_node = false, priority = TaskPriority.low,
                   args = (), kwargs...)

Parallel map-reduce operation: applies `map_func` to `iterables` and then reduces the results with `reduce_func`

# Arguments
- `map_func`: function applied to each element (or tuple of elements) of `iterables`
- `reduce_func`: associative function to combine the mapped results
- `iterables...`: one or more iterables of input data

# Keyword Arguments
- `chunksize`: number of elements per task (auto-calculated if `nothing`)
- `direction`: `:column` or `:row` for reduction  
- `mode`: `"1d"` or `"2d"` reduction mode  
- `qid`: target MPI node (default: round-robin)
- `counted`: track spawned tasks in runtime statistics
- `forced_node`: pin tasks to the specified `qid`
- `priority`: tasks priority
- `args`, `kwargs`: extra positional or keyword arguments passed to both `map_func` and `reduce_func`

# Returns
Reduced value computed from applying `map_func` over the input data and
aggregating with `reduce_func`

# Example
data = [1, 2, 3, 4]
mapReduce(x -> x^2, +, data)  # returns 30 (sum of squares)
"""
function mapReduce(map_func, reduce_func, iterables...; chunksize::Union{Int, Nothing} = nothing, 
                            direction::Symbol = :column, mode::String = "1d", qid::Int = -1, 
                            counted::Bool = true, forced_node::Bool = false, priority::TaskPriorityLevel = TaskPriority.low, 
                            args::Tuple = (), kwargs...)
    chunksize = something(chunksize, max(cld(length(iterables[1]), ctx[].num_procs * TORC_NUM_WORKERS), 1))

    # 1 -> map
    map_items = map(map_func, iterables...; chunksize = chunksize, qid = qid, counted = counted, forced_node = forced_node,
                         priority = priority, args = args, kwargs...)
    # 2 -> reduce
    reduce(reduce_func, map_items; chunksize = chunksize, direction = direction, mode = mode, qid = qid, counted = counted,
                forced_node = forced_node, priority = priority, args = args, kwargs...)
end

"""
    filter(func, iterables...; chunksize = nothing, qid = -1,
                counted = true, forced_node = false, priority = TaskPriority.low,
                args = (), kwargs...)

Parallel version of Python's `filter()`: returns elements of `iterables` for which `func` returns `true`

# Arguments
- `func`: function returning `true` for elements to keep
- `iterables...`: one or more iterables of input data

# Keyword Arguments
- `chunksize`: number of elements per task (auto-calculated if `nothing`)
- `qid`: target MPI node (default: round-robin)
- `counted`: track spawned tasks in runtime statistics
- `forced_node`: pin tasks to the specified `qid`
- `priority`: tasks priority
- `args`, `kwargs`: extra positional or keyword arguments passed to `func`

# Returns
A collection of the same type as the input, containing only the elements that satisfy `func`

# Example
a = [1, 2, 3, 4]
filter(x -> x % 2 == 0, a)  # returns [2, 4]
"""
function filter(func, iterables...; chunksize::Union{Int, Nothing} = nothing,  qid::Int = -1, 
                            counted::Bool = true, forced_node::Bool = false, priority::TaskPriorityLevel = TaskPriority.low, 
                            args::Tuple = (), kwargs...)

    items = zip(iterables...)
    T = eltype(iterables[1])
   
    _constr = typeof(iterables[1]) <: AbstractArray ? collect : typeof(iterables[1])

    chunksize = something(chunksize, max(cld(length(iterables[1]), ctx[].num_procs * TORC_NUM_WORKERS), 1))

    _wrapper = chunksize == 1 ? (x -> func(x..., args...; kwargs...) ? Some(T(x[1])) : nothing) :
        (ch -> begin out = Vector{T}(); for x in ch func(x..., args...; kwargs...) && push!(out, T(x[1])); end; out end)

    futures = chunksize == 1 ? [submit(_wrapper, x; qid = qid, counted = counted, forced_node = forced_node, priority = priority) for x in items] :
                               [submit(_wrapper, ch; qid = qid, counted = counted, forced_node = forced_node, priority = priority) 
                                        for ch in Iterators.partition(items, chunksize)]
                
    _waitall(false)

    results = [result(t) for t in futures]
    _constr(chunksize == 1 ? T[something(x) for x in raw_results if x !== nothing] : Base.reduce(vcat, results; init = T[]))
end

"""
    mapPairs(func, xs, ys; chunksize=nothing, qid = -1, counted = true, forced_node = false, priority = TaskPriority.low, args = (), kwargs...)

Apply `func` to all pairs `(x, y)` from vectors `xs` and `ys` in parallel

# Keyword Arguments
- `chunksize`: number of index pairs per task (auto-calculated if `nothing`)  
- `qid`: target MPI node (default: round-robin)  
- `counted`: track spawned tasks in runtime statistics  
- `forced_node`: pin tasks to the specified qid  
- `priority`: tasks priority  
- `args`, `kwargs`: extra positional or keyword arguments passed to `func`  

# Returns
Matrix of size `length(xs) * length(ys)` containing `func(x, y)` for each `(x, y)` pair

# Example
xs = [1, 2]
ys = [10, 20]
mapPairs((x, y) -> x + y, xs, ys)  # returns [11 21; 12 22]
"""
function mapPairs(func::Function, xs::AbstractVector, ys::AbstractVector; chunksize::Union{Int, Nothing} = nothing, qid::Int = -1, 
                       counted::Bool = true, forced_node::Bool = false, priority::TaskPriorityLevel = TaskPriority.low, 
                       args::Tuple = (), kwargs...)
    nx, ny = length(xs), length(ys)

    chunksize = something(chunksize, max(cld(length(xs) * length(ys), ctx[].num_procs * TORC_NUM_WORKERS), 1))
    
    idx_pairs = Iterators.product(1:nx, 1:ny)
    chunks = chunksize == 1 ? [(i,) for i in idx_pairs] : Iterators.partition(idx_pairs, chunksize)

    futures = [submit(ch -> [func(xs[i], ys[j], args...; kwargs...) for (i,j) in ch], ch; qid = qid, 
                      counted = counted, forced_node = forced_node, priority = priority) for ch in chunks]
    _waitall(false)

    partials = Iterators.flatten(result.(futures))
    T = typeof(first(partials)); results = Array{T}(undef, nx, ny)

    for (val, (i,j)) in zip(Iterators.flatten(result.(futures)), idx_pairs)
        results[i,j] = val
    end

    results
end

"""
    mapPairsReduce(map_func, reduce_func, xs, ys; chunksize = nothing, direction = :column, mode = "1d", qid = -1,
                        counted = true, forced_node = false, priority = TaskPriority.low, args = (), kwargs...)

Parallel map-pairs followed by reduction: applies `map_func` to all `(x, y)` pairs from `xs` and `ys`, then reduces the results with `reduce_func`

# Arguments
- `map_func`: function applied to each pair `(x, y)`  
- `reduce_func`: associative function to combine the mapped results  
- `xs`, `ys`: input vectors whose Cartesian product defines the pairs  

# Keyword Arguments
- `chunksize`: number of pairs per task (auto-calculated if `nothing`)  
- `direction`: `:column` or `:row` for reduction  
- `mode`: `"1d"` or `"2d"` reduction mode  
- `qid`: target MPI node (default: round-robin)  
- `counted`: track spawned tasks in runtime statistics  
- `forced_node`: pin tasks to the specified `qid`  
- `priority`: tasks priority  
- `args`, `kwargs`: extra positional or keyword arguments passed to both `map_func` and `reduce_func`  

# Returns
Reduced result computed from applying `map_func` over all `(x, y)` pairs and aggregating with `reduce_func`

# Example
xs = [1, 2]
ys = [10, 20]
mapPairsReduce((x, y) -> x*y, +, xs, ys)  # returns 90 (1*10 + 1*20 + 2*10 + 2*20)
"""
function mapPairsReduce(map_func::Function, reduce_func::Function, xs::AbstractVector, ys::AbstractVector; chunksize::Union{Int, Nothing} = nothing, 
                            direction::Symbol = :column, mode::String = "1d", qid::Int = -1, counted::Bool = true, forced_node::Bool = false, 
                            priority::TaskPriorityLevel = TaskPriority.low, args::Tuple = (), kwargs...)

    chunksize = something(chunksize, max(cld(length(xs) * length(ys), ctx[].num_procs * TORC_NUM_WORKERS), 1))
    
    # 1 -> pairs from map
    mapPair_mat = mapPairs(map_func, xs, ys; chunksize = chunksize, qid = qid, counted = counted,
                                forced_node = forced_node, priority = priority, args = args, kwargs...)
    # 2 -> reduce
    reduce(reduce_func, mapPair_mat; chunksize = chunksize, direction = direction, mode = mode, qid = qid, counted = counted,
                forced_node = forced_node, priority = priority, args = args, kwargs...)
end

"""
    scan(func, data; chunksize = nothing, mode = "inclusive", identity = nothing, qid = -1,
              counted = true, forced_node = false, priority = TaskPriority.low, args = (), kwargs...)

Compute the parallel prefix scan (cumulative reduction) of a vector `data` using the associative function `func`

# Keyword Arguments
- `chunksize`: number of elements per task (auto-calculated if `nothing`)  
- `mode`: `"inclusive"` (default) or `"exclusive"` scan  
- `identity`: identity value for `"exclusive"` mode (default: zero of element type)  
- `qid`: target MPI node (default: round-robin)  
- `counted`: track spawned tasks in runtime statistics  
- `forced_node`: pin tasks to the specified `qid`  
- `priority`: task priority  
- `args`, `kwargs`: extra positional or keyword arguments passed to `func`  

# Returns
Vector of the same length as `data` containing the cumulative results of `func`

# Example
data = [1, 2, 3, 4]
scan(+, data)
# => [1, 3, 6, 10]  # inclusive sum scan

scan(+, data, mode = "exclusive")
# => [0, 1, 3, 6]   # exclusive sum scan, using zero as identity
"""
function scan(func::Function, data::AbstractVector; chunksize::Union{Int, Nothing} = nothing, mode::String = "inclusive", 
                   identity::Union{Any, Nothing} = nothing, qid::Int = -1, counted::Bool = true, forced_node::Bool = false, 
                   priority::TaskPriorityLevel = TaskPriority.low, args::Tuple = (), kwargs...)
    
     if !(mode in ("inclusive", "exclusive"))
        @error "mode must be \"inclusive\" or \"exclusive\""
        MPI_mod.Abort(ctx[].comm, 1)
    end
    
    T = eltype(data)

    if identity !== nothing && !(identity isa T)
        @error "identity type must match data element type"
        MPI_mod.Abort(ctx[].comm, 1)
    end

    ndim = length(data)
    chunksize = something(chunksize, max(cld(ndim, ctx[].num_procs * TORC_NUM_WORKERS), 1))

    prefix_chunk = (x, args...; kwargs...) -> begin
        _x = collect(x)
        res = similar(_x)
        if mode == "exclusive"
            res[1] = identity === nothing ? zero(T) : identity
            for i in 2:length(_x)
                res[i] = func(res[i-1], _x[i-1], args...; kwargs...)
            end
        else 
            res[1] = _x[1]
            for i in 2:length(_x)
                res[i] = func(res[i-1], _x[i], args...; kwargs...)
            end
        end
        res
    end
    
    _wrapper_prefix = x -> prefix_chunk(x, args...; kwargs...)
    
    futures = [submit(_wrapper_prefix, ch; qid = qid, counted = counted, forced_node = forced_node, priority = priority) 
               for ch in Iterators.partition(data, chunksize)]
    _waitall(false)
    
    chunk_pref = [result(f) for f in futures]

    _chunk_acc = [c[end] for c in chunk_pref[1:end - 1]]
    _total_acc = similar(_chunk_acc)
    _total_acc[1] = _chunk_acc[1]
    
    if mode == "inclusive"
        for i in 2:length(_chunk_acc)
            _total_acc[i] = func(_total_acc[i-1], _chunk_acc[i], args...; kwargs...)
        end
    else
        _total_acc[1] = func(_total_acc[1], data[chunksize], args...; kwargs...)

        for i in 2:length(_total_acc)
            tmp = func(_total_acc[i-1], _chunk_acc[i], args...; kwargs...)
            _total_acc[i] = func(tmp, data[i * chunksize], args...; kwargs...)
        end
    end

    _reduce_chunks = (c, offset, args...; kwargs...) -> begin
        _x = collect(c); [func(x, offset, args...; kwargs...) for x in _x]
    end

    _wrapper_reduce = (x, offset) -> _reduce_chunks(x, offset, args...; kwargs...)
    
    futures = [submit(_wrapper_reduce, ch, off; qid = qid, counted = counted, forced_node = forced_node, priority = priority)
               for (ch, off) in zip(chunk_pref[2:end], _total_acc)]
    _waitall(false) 
    
    results = vcat(chunk_pref[1], [result(t) for t in futures])
    
    vcat(results...)
end

"""
    shm_alloc(dim::Tuple{Vararg{Int}}, T::Type)

Allocate a shared memory array of type `T` with dimensions `dim` on all MPI nodes
"""
function shm_alloc(dim::Tuple{Vararg{Int}}, T::Type)
    for rank in Base.filter(r -> r != Int(ctx[].rank), 0:(Int(ctx[].num_procs) - 1))
        submit(_alloc_shared_memwin, dim, T, ctx[].rank; qid = rank, forced_node = true, counted = false)
    end

    mem = _alloc_shared_memwin(dim, T, ctx[].rank)
    _waitall(false)

    mem
end

"""
    _alloc_shared_memwin(size::Tuple{Vararg{Int}}, type::Type, master::UInt)

Allocate a shared memory window of the given type and size on the master node, 
allowing other nodes to access it via MPI shared memory
"""
function _alloc_shared_memwin(size::Tuple{Vararg{Int}}, type::Type, master::UInt)
    win, arr = MPI_mod.Win_allocate_shared(Array{type}, (ctx[].rank == master ? size : (0,)), ctx[].comm)
    ctx[].rank != master ? arr = MPI_mod.Win_shared_query(Array{type}, size, win; rank = master) : nothing

    ptr = ctx[].rank == master ? UInt(pointer_from_objref(arr)) : UInt(0)
    ptr = MPI_mod.bcast(ptr, ctx[].comm; root = master)

    _reg_shared_mem(ptr, arr, win)

    arr
end

"""
    _reg_shared_mem(ptr::UInt, buffer, win::MPI_mod.Win)

Register a shared memory window in shm registry
"""
_reg_shared_mem(ptr::UInt, buffer, win::MPI_mod.Win) = torc_shm[ptr] = (buffer, win)

"""
    _get_shared_mem(ptr::UInt)

Retrieve the shared memory array associated with the given pointer
"""
_get_shared_mem(ptr::UInt) = haskey(torc_shm, ptr) ? torc_shm[ptr][1] : nothing

"""
    _set_scheduling_policy(policy::Symbol, weights::Union{Vector{Int}, Nothing})

Configure the task scheduling policy across all MPI nodes

`policy`: scheduling policy (`:ROUND_ROBIN`, `:WRR`, `:LOCAL`, `:RANDOM`)
`weights`: node weights for weighted scheduling (required for `:WRR`)

# Note
- Validation errors trigger automatic fallback to `:ROUND_ROBIN` with warning messages
"""
function _set_scheduling_policy(policy::Symbol, weights::Union{Vector{Int}, Nothing})
    if policy == :WRR
        if isnothing(weights)
            @warn "Node weights not provided for Weighted Round-Robin (WRR) -> falling back to Round-Robin"
            policy = :ROUND_ROBIN
        elseif length(weights) != ctx[].num_procs
            @warn "Weights size ($(length(weights))) doesn't match the number of nodes ($(ctx[].num_procs)) -> falling back to Round-Robin"
            policy = :ROUND_ROBIN
        else
            spmd(() -> begin
                TORC_DISTRIBUTION_POLICY[] = _parse_policy(policy)
            
                node_weights[] = weights
                node_weights_cumsum[] = cumsum(node_weights[]) 
            end, counted = false, forced_node = true)
        end
    end

    spmd(() -> begin
        TORC_DISTRIBUTION_POLICY[] = _parse_policy(policy)
    end, counted = false, forced_node = true)
end

function _enable_stealing()
    spmd(() -> (TORC_STEALING_ENABLED[] = true), priority = TaskPriority.high, counted = false, forced_node = true)
end

function _disable_stealing()
    spmd(() -> (TORC_STEALING_ENABLED[] = false), priority = TaskPriority.high, counted = false, forced_node = true)
end

function gettime()
    time_ns() * 1e-9
end
    
end