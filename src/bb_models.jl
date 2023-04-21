export BBModel, obj, obj_nomad, obj_cat, cost

"""Mutable struct `BBModel`

Represents a black box optimization problem that follows the NLPModel API.

The following constructors are available:

    BBModel(parameter_set, problems, solver_function, f; kwargs...)
    BBModel(parameter_set, problems, solver_function, f, c, lcon, ucon; kwargs...)

- `parameter_set::AbstractParameterSet`: structure containing parameters information;
- `problems::Vector{AbstractNLPModel}`: set of problem to run the benchmark on;
- `solver_function::Function`: function that takes an `AbstractNLPModel` and a `AbstractParameterSet` and returns a [`GenericExecutionStats`](https://github.com/JuliaSmoothOptimizers/SolverCore.jl/blob/main/src/stats.jl).
- `f::Function`: Given a `Vector{ProblemMetrics}` returns a score as a Float64 (examples are [`time_only`](@ref), [`memory_only`](@ref), [`sumfc`](@ref));

For constrained problems:

    lcon ≤ c(x) ≤ ucon

- `c::Function`: constraint function;
- `lcon::AbstractVector`: lower bound on the constraint;
- `ucon::AbstractVector`: upper bound on the constraint.

Additional keyword arguments are:

- `subset::NTuple{N, Symbol}`: subset of parameters to be considered (by default all parameters from `parameter_set`);
- `x0::AbstractVector`: initial values for the parameters (by default `Float64.(values(parameter_set))`);
- `lvar::AbstractVector`: lower bound on the the parameters (by default `Float64.(lower_bounds(parameter_set))`);
- `uvar::AbstractVector`: upper bound on the the parameters (by default `Float64.(lower_bounds(parameter_set))`);
- `name::String`: name of the problem (by default: "Generic").

Note that if `x0` is not provided, the computations are run in `Vector{Float64}`.
"""
mutable struct BBModel{
  P <: AbstractParameterSet,
  T,
  S,
  F1 <: Function,
  F2 <: Function,
  F3 <: Function,
  N,
} <: AbstractNLPModel{T, S}
  bb_meta::BBModelMeta
  meta::NLPModelMeta{T, S}
  counters::Counters
  solver_function::F1
  f::F2
  c::F3
  problems::Dict{Int, Problem}
  parameter_set::P
  subset::NTuple{N, Symbol}
end

NLPModels.show_header(io::IO, ::BBModel) = println(io, "BBModel - Black Box Optimization Model")

function BBModel(
  parameter_set::P,
  problems::Vector{M},
  solver_function::Function,
  f::Function = time_only;
  subset::NTuple{N, Symbol} = fieldnames(P),
  x0::S = Float64.(values_num(subset, parameter_set)),
  lvar::S = eltype(x0).(lower_bounds(subset, parameter_set)),
  uvar::S = eltype(x0).(upper_bounds(subset, parameter_set)),
  name::String = "generic-BBModel",
) where {M <: Union{AbstractNLPModel, Function}, S, N, P <: AbstractParameterSet}
  length(problems) > 0 || error("No problems given")
  nvar = length(x0)
  @lencheck nvar lvar uvar
  bbmeta = BBModelMeta(parameter_set, subset)
  meta = NLPModelMeta(nvar; x0 = x0, lvar = lvar, uvar = uvar, name = name)
  problems = Dict{Int, Problem}(id => Problem(id, p, eps()) for (id, p) ∈ enumerate(problems))
  return BBModel(
    bbmeta,
    meta,
    Counters(),
    solver_function,
    f,
    x -> Float64[],
    problems,
    parameter_set,
    subset,
  )
end

function BBModel(
  parameter_set::P,
  problems::Vector{M},
  solver_function::Function,
  f::Function,
  c::Function,
  lcon::S,
  ucon::S;
  subset::NTuple{N, Symbol} = fieldnames(P),
  x0::S = eltype(S).(values_num(subset, parameter_set)),
  lvar::S = eltype(S).(lower_bounds(subset, parameter_set)),
  uvar::S = eltype(S).(upper_bounds(subset, parameter_set)),
  name::String = "generic-BBModel",
) where {M <: Union{AbstractNLPModel, Function}, S, N, P <: AbstractParameterSet}
  length(problems) > 0 || error("No problems given")
  nvar, ncon = length(x0), length(lcon)
  @lencheck ncon ucon
  @lencheck nvar lvar uvar
  bbmeta = BBModelMeta(parameter_set, subset)
  meta = NLPModelMeta(
    nvar;
    x0 = x0,
    ncon = ncon,
    lvar = lvar,
    uvar = uvar,
    lcon = lcon,
    ucon = ucon,
    name = name,
  )
  problems = Dict{Int, Problem}(id => Problem(id, p, eps()) for (id, p) ∈ enumerate(problems))
  return BBModel(bbmeta, meta, Counters(), solver_function, f, c, problems, parameter_set, subset)
end

"""
    obj(nlp::BBModel, x::AbstractVector; kwargs...)

Objective function of the `BBModel`. 
The difference with [`obj_cat`](@ref) is that `x` contains only the numerical parameters (excluding categorical variables).
Therefore, `x` is of length `nlp.meta.nvar`.

The keyword arguments are passed to [`cost`](@ref).
"""
function NLPModels.obj(nlp::BBModel, x::AbstractVector; kwargs...)
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_obj)
  param_set, subset = nlp.parameter_set, nlp.subset
  set_values_num!(subset, param_set, x)

  # Support parallel processes
  futures = Dict{Int, Future}()
  @sync for worker_id in workers()
    @async futures[worker_id] = @spawnat worker_id let worker_metrics = Vector{ProblemMetrics}(undef, length(PROBLEM_PARTITION))
      # global variable declared in BBModels.jl
      # Contains the id of the problems. 
      global PROBLEM_PARTITION
      for (i, pb_id) in enumerate(PROBLEM_PARTITION)
        worker_metrics[i] = cost(nlp, nlp.problems[pb_id]; kwargs...)
      end
      return worker_metrics
    end
  end
  # Collect all the metrics from the workers
  worker_metrics = Dict{Int, Vector{ProblemMetrics}}()
  @sync for worker_id in keys(futures)
    @async worker_metrics[worker_id] = fetch(futures[worker_id])
  end

  vec_metric = vcat(collect(values(worker_metrics))...)
  # return worker_metrics to get data from workers
  return nlp.f(vec_metric), worker_metrics
end

"""
    obj_nomad(nlp::BBModel, x::AbstractVector, problems::Vector{Problem}; kwargs...)

Objective function of the `BBModel` to be used with SolverTuning.jl. 
The difference with [`obj`](@ref) is the positional parameter `problems`.
Therefore, `x` is of length `nlp.meta.nvar`.

The keyword arguments are passed to [`cost`](@ref).
"""
function obj_nomad(nlp::BBModel, x::AbstractVector, problems::Vector{Problem}; kwargs...)
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_obj)
  param_set, subset = nlp.parameter_set, nlp.subset
  vec_metric = Vector{ProblemMetrics}(undef, length(problems))
  set_values_num!(subset, param_set, x)
  for (i, problem) in enumerate(problems)
    vec_metric[i] = cost(nlp, problem; kwargs...)
  end
  return nlp.f(vec_metric), vec_metric
end
"""
    obj_cat(nlp::BBModel, x::AbstractVector; kwargs...)


Objective function of the `BBModel`.
The difference with [`obj`](@ref) is that `x` contains all the parameters (including categorical variables).
Therefore, `x` is of length `nlp.bb_meta.nvar`.

The keyword arguments are passed to [`cost`](@ref).
"""
function obj_cat(nlp::BBModel, x::AbstractVector; kwargs...)
  @lencheck nlp.bb_meta.nvar x
  increment!(nlp, :neval_obj)
  param_set, subset = nlp.parameter_set, nlp.subset
  vec_metric = Vector{ProblemMetrics}(undef, length(nlp.problems))
  for (pb_id, problem) in nlp.problems
    set_values!(subset, param_set, x)
    vec_metric[pb_id] = cost(nlp, problem; kwargs...)
  end
  return nlp.f(vec_metric)
end

"""
    cost(nlp::BBModel, p::Problem; seconds = 10.0, samples = 1, evals = 1)

For a given problem `p::Problem`, it returns a [`ProblemMetrics`](@ref) containing the benchmark's results of `nlp.solver_function`.
The keyword arguments are parameters for the benchmark.
"""
function cost(nlp::BBModel, p::Problem; seconds = 10.0, samples = 1, evals = 1)
  haskey(nlp.problems, get_id(p)) || error("Problem could not be found in problem set")

  solver_function = nlp.solver_function
  param_set = nlp.parameter_set
  nlp_to_solve = get_nlp(p)
  bmark_result, stat =
    @benchmark_with_result $solver_function($nlp_to_solve, $param_set) seconds = seconds samples =
      samples evals = evals

  times = bmark_result.times
  normalize_times!(times)
  memory = bmark_result.memory
  counters = deepcopy(nlp_to_solve.counters)
  p_metric = ProblemMetrics(get_id(p), times, memory, stat.status, counters)

  reset!(nlp_to_solve)

  return p_metric
end

function NLPModels.cons!(nlp::BBModel, x::AbstractVector, c::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon c
  increment!(nlp, :neval_cons)
  c .= nlp.c(x)
  return c
end

function NLPModels.reset_data!(nlp::BBModel)
  problem_collection = (get_nlp(p) for p ∈ nlp.problems)
  for p ∈ problem_collection
    reset!(p)
  end

  return nlp
end
