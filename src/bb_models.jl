export BBModel, obj, obj!

"""Mutable struct `BBModel`

Represents a black box optimization problem that follows the NLPModel API.

The following constructors are available:

    BBModel(parameter_set, problems, solver_function, auxiliary_function; kwargs...)
    BBModel(parameter_set, problems, solver_function, auxiliary_function, c, lcon, ucon; kwargs...)

- `parameter_set::AbstractParameterSet`: structure containing parameters information;
- `problems::Vector{AbstractNLPModel}`: set of problem to run the benchmark on;
- `solver_function::Function`: function that takes an `AbstractNLPModel` and a `AbstractParameterSet` and returns a [`GenericExecutionStats`](https://github.com/JuliaSmoothOptimizers/SolverCore.jl/blob/main/src/stats.jl).
- `auxiliary_function::Function`: Given a `ProblemMetrics` returns a score as a Float64 (examples are `time_only`, `memory_only`, `sumfc`);

For constrained problems:

    lcon ≤ c(x) ≤ ucon

- `c::Function`: constraint function;
- `lcon::AbstractVector`: lower bound on the constraint;
- `ucon::AbstractVector`: upper bound on the constraint.

Additional keyword arguments are:

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
} <: AbstractNLPModel{T, S}
  bb_meta::BBModelMeta
  meta::NLPModelMeta{T, S}
  counters::Counters
  solver_function::F1
  auxiliary_function::F2
  c::F3
  problems::Dict{Int, Problem}
  parameter_set::P
end

NLPModels.show_header(io::IO, ::BBModel) = println(io, "BBModel - Black Box Optimization Model")

function BBModel(
  parameter_set::AbstractParameterSet,
  problems::Vector{M},
  solver_function::Function,
  auxiliary_function::Function = time_only;
  x0::S = Float64.(values(parameter_set)),
  lvar::S = eltype(x0).(lower_bounds(parameter_set)),
  uvar::S = eltype(x0).(upper_bounds(parameter_set)),
  name::String = "generic-BBModel",
) where {M <: AbstractNLPModel, S}
  length(problems) > 0 || error("No problems given")
  nvar = length(x0)
  @lencheck nvar lvar uvar
  bbmeta = BBModelMeta(parameter_set)
  meta = NLPModelMeta(nvar; x0 = x0, lvar = lvar, uvar = uvar, name = name)
  problems = Dict{Int, Problem}(id => Problem(id, p, eps()) for (id, p) ∈ enumerate(problems))
  return BBModel(
    bbmeta,
    meta,
    Counters(),
    solver_function,
    auxiliary_function,
    x -> T[],
    problems,
    parameter_set,
  )
end

function BBModel(
  parameter_set::AbstractParameterSet,
  problems::Vector{M},
  solver_function::Function,
  auxiliary_function::Function,
  c::Function,
  lcon::S,
  ucon::S;
  x0::S = eltype(S).(values(parameter_set)),
  lvar::S = eltype(S).(lower_bounds(parameter_set)),
  uvar::S = eltype(S).(upper_bounds(parameter_set)),
  name::String = "generic-BBModel",
) where {M <: AbstractNLPModel, S}
  length(problems) > 0 || error("No problems given")
  nvar, ncon = length(x0), length(lcon)
  @lencheck ncon ucon
  @lencheck nvar lvar uvar
  bbmeta = BBModelMeta(parameter_set)
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
  return BBModel(
    bbmeta,
    meta,
    Counters(),
    solver_function,
    auxiliary_function,
    c,
    problems,
    parameter_set,
  )
end

# By default, this function will return the time in seconds
function NLPModels.obj(nlp::BBModel, x::AbstractVector)
  problems = nlp.problems
  solver_function = nlp.solver_function
  total_time = 0.0
  for (pb_id, problem) in problems
    total_time += @elapsed solver_function(get_nlp(problem), x)
  end

  return total_time
end

# Function to use for NOMAD: assumes that an interface will sanitize Nomad's output
function obj!(nlp::BBModel, p::Problem)
  haskey(nlp.problems, get_id(p)) || error("Problem could not be found in problem set")

  solver_function = nlp.solver_function
  auxiliary_function = nlp.auxiliary_function
  nlp_to_solve = get_nlp(p)
  param_set = nlp.parameter_set
  bmark_result, stat =
    @benchmark_with_result $solver_function($nlp_to_solve, $param_set) seconds = 10 samples = 5 evals =
      1
  times = bmark_result.times
  normalize_times!(times)
  memory = bmark_result.memory
  solved = !is_failure(stat)
  counters = deepcopy(nlp_to_solve.counters)
  p_metric = ProblemMetrics(get_id(p), times, memory, solved, counters)
  reset!(nlp_to_solve)

  return auxiliary_function(p_metric), p_metric
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
