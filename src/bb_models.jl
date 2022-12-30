export AbstractBBModel, BBModel, obj

abstract type AbstractBBModel{T, S} <: AbstractNLPModel{T, S} end

"""Mutable struct BBModel

Represents a black box optimization problem that follows the api described in `NLPModels`.

`solver_function` is a function that takes an `AbstractNLPModel` and a `AbstractParameterSet` and runs a solver with thise two inputs.

  Ex: ```julia
  solver_function(nlp::AbstractNLPModel, p::AbstractParameterSet) = solve!(nlp, p;)  
  ```
`auxiliary_function` is a function that takes `ProblemMetrics` as a parameter and return a `Vector{Float64}`.
  This function is the objective function that needs to be minimized/maximized as it represents how well your solver performed while solving a given problem.
    Ex: ```julia
    function aux_func(p_metric::ProblemMetrics)
      T = Float64
      median_time = T(median(get_times(p_metric)))
      memory = T(get_memory(p_metric))
      is_not_solved = T(!get_solved(p_metric))
      counters = get_counters(p_metric)
      return T[median_time + memory + T(counters.neval_obj) + (is_not_solved * median_time)]
    end
    ``` 
  `c` is a function returning a `Vector{Float64}` that represents the constraints of the problem.
  `problems` is a dictionary containing `AbstractNLPModels` to solve using the `solver_function` method.
  `parameter_set` is the `AbstractParameterSet` of a given solver.
"""
mutable struct BBModel{F1 <: Function, F2 <: Function, P <: AbstractParameterSet, T, S} <:
               AbstractBBModel{T, S}
  bb_meta::BBModelMeta
  meta::NLPModelMeta{T, S}
  counters::Counters
  solver_function::F1
  auxiliary_function::F2
  c
  problems::Dict{Int, Problem}
  parameter_set::P
end

NLPModels.show_header(io::IO, ::BBModel) = println(io, "BBModel - Black Box Optimization Model")

function BBModel(
  parameter_set::P,
  solver_function::F1,
  auxiliary_function::F2,
  problems::Vector{M};
  lvar = lower_bounds(parameter_set),
  uvar = upper_bounds(parameter_set),
  kwargs...,
) where {P <: AbstractParameterSet, F1 <: Function, F2 <: Function, M <: AbstractNLPModel}
  x_n = names(parameter_set)
  return BBModel(
    values(parameter_set),
    solver_function,
    auxiliary_function,
    problems,
    parameter_set,
    x_n,
    lvar,
    uvar;
    kwargs...,
  )
end

function BBModel(
  x0::AbstractVector,
  solver_function::F1,
  auxiliary_function::F2,
  problems::Vector{M},
  parameter_set::P,
  x_n::Vector{String},
  lvar::AbstractVector,
  uvar::AbstractVector;
  name::String = "generic-BBModel",
) where {P <: AbstractParameterSet, F1 <: Function, F2 <: Function, M <: AbstractNLPModel}
  length(problems) > 0 || error("No problems given")
  nvar = length(x0)
  bbmeta = BBModelMeta(nvar, x0, x_n)
  meta_x0 = Vector{Float64}([Float64(i) for i in x0])
  meta_lvar = Vector{Float64}([Float64(i) for i in lvar])
  meta_uvar = Vector{Float64}([Float64(i) for i in uvar])
  meta = NLPModelMeta(nvar; x0 = meta_x0, lvar = meta_lvar, uvar = meta_uvar, name = name)
  problems =
    Dict{Int, Problem}(id => Problem(id, p, eps(Float64)) for (id, p) ∈ enumerate(problems))
  return BBModel(
    bbmeta,
    meta,
    Counters(),
    solver_function,
    auxiliary_function,
    x -> Float64[],
    problems,
    parameter_set,
  )
end

# Constructor with constraints
function BBModel(
  parameter_set::P,
  solver_function::F1,
  auxiliary_function::F2,
  c::Function,
  lcon::Vector{Float64},
  ucon::Vector{Float64},
  problems::Vector{M};
  lvar = lower_bounds(parameter_set),
  uvar = upper_bounds(parameter_set),
  kwargs...,
) where {P <: AbstractParameterSet, F1 <: Function, F2 <: Function, M <: AbstractNLPModel}
  x_n = names(parameter_set)
  x0 = values(parameter_set)
  return BBModel(
    x0,
    solver_function,
    auxiliary_function,
    c,
    lcon,
    ucon,
    problems,
    parameter_set,
    x_n,
    lvar,
    uvar;
    kwargs...,
  )
end

function BBModel(
  x0::AbstractVector,
  solver_function::F1,
  auxiliary_function::F2,
  c::Function,
  lcon::Vector{Float64},
  ucon::Vector{Float64},
  problems::Vector{M},
  parameter_set::P,
  x_n::Vector{String},
  lvar::AbstractVector,
  uvar::AbstractVector;
  name::String = "generic-BBModel",
) where {P <: AbstractParameterSet, F1 <: Function, F2 <: Function, M <: AbstractNLPModel}
  length(problems) > 0 || error("No problems given")
  @lencheck ncon ucon lcon
  nvar = length(x0)
  bbmeta = BBModelMeta(nvar, x0, x_n)
  meta = NLPModelMeta(
    nvar;
    x0 = Vector{Float64}([Float64(i) for i in x0]),
    lvar = lvar,
    uvar = uvar,
    name = name,
  )
  problems =
    Dict{Int, Problem}(id => Problem(id, p, eps(Float64)) for (id, p) ∈ enumerate(problems))

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

function NLPModels.obj(nlp::BBModel, x::AbstractVector; seconds = 10.0, samples = 1, evals = 1)
  @lencheck nlp.bb_meta.nvar x
  param_set = nlp.parameter_set
  total = 0.0
  for (pb_id, problem) in nlp.problems
    SolverParameters.set_values_num!(param_set, x)
    p_metric = cost(nlp, problem, seconds = seconds, samples = samples, evals = evals)
    total += nlp.auxiliary_function(p_metric)
  end

  return total
end

function cost(nlp::BBModel, p::Problem; seconds = 10.0, samples = 1, evals = 1)
  haskey(nlp.problems, get_id(p)) || error("Problem could not be found in problem set")

  solver_function = nlp.solver_function
  param_set = nlp.parameter_set
  nlp_to_solve = get_nlp(p)
  bmark_result, stat = @benchmark_with_result $solver_function($nlp_to_solve, $param_set) seconds = seconds samples = samples evals = evals

  times = bmark_result.times
  normalize_times!(times)
  memory = bmark_result.memory
  solved = !is_failure(stat)
  counters = deepcopy(nlp_to_solve.counters)
  p_metric = ProblemMetrics(get_id(p), times, memory, solved, counters)

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
