export AbstractBBModel, BBModel, obj, obj!

abstract type AbstractBBModel{T, S} <: AbstractNLPModel{T, S} end

mutable struct BBModel{T, S <: AbstractVector{<:Real}, P, F1 <: Function, F2 <: Function} <: AbstractBBModel{T, S}
  meta::BBModelMeta{T,S}
  counters::Counters

  solver_function::F1
  auxiliary_function::F2
  problems::Dict{Int, Problem{P}}
  bb_results::Vector{Vector{ProblemMetrics}}

  function BBModel(meta::BBModelMeta{T,S}, counters::Counters, s_f::F1, a_f::F2, problems::Dict{Int, Problem{P}}, bb_results::Vector{Vector{ProblemMetrics}}
    ) where {T, S <: AbstractVector{<:Real}, P, F1 <: Function, F2 <: Function}
    new{T, S, P, F1, F2}(meta, counters, s_f, a_f, problems, bb_results)
  end
end

NLPModels.show_header(io::IO, ::BBModel) = println(io, "BBModel - Black Box Optimization Model")

# TODO: create a type for the constraints...
function BBModel(x0::NamedTuple, solver_function::F1, auxiliary_function::F2, problems::Vector{M};
  kwargs...
  ) where {F1 <: Function, F2 <: Function, M <: AbstractNLPModel}
  x_n = collect(keys(x0))
  T = Union{(typeof(tᵢ) for tᵢ in x0)...}
  x0 = collect(T, x0)
  kwargs = Dict(kwargs)
  
  if haskey(kwargs, :lvar)
    lvar = T[convert(x_tᵢ, l) for (x_tᵢ, l) in zip((typeof(x0ᵢ) for x0ᵢ in x0), kwargs[:lvar])]
    delete!(kwargs, :lvar)
  end
  if haskey(kwargs, :uvar)
    uvar = T[convert(x_tᵢ, l) for (x_tᵢ, l) in zip((typeof(x0ᵢ) for x0ᵢ in x0), kwargs[:uvar])]
    delete!(kwargs, :uvar)
  end
  
  return BBModel(x0, solver_function, auxiliary_function, problems; x_n=x_n, lvar=lvar, uvar=uvar, kwargs...)
  # return BBModel(collect(T, x0), solver_function, auxiliary_function, problems; x_n=x_n, lvar=lvar, uvar=uvar, lcon=lcon, ucon=ucon, kwargs...)
end

function BBModel(
  x0::S,
  solver_function::F1,
  auxiliary_function::F2,
  problems::Vector{M};
  x_n::Vector{Symbol}=Symbol[Symbol("param_", i) for i in 1:length(x0)],
  lvar::S = eltype(S)[typemin(typeof(x0ᵢ)) for x0ᵢ in x0],
  uvar::S = eltype(S)[typemax(typeof(x0ᵢ)) for x0ᵢ in x0],
  lcon::Vector{Float64} = Vector{Float64}(undef, 0),
  ucon::Vector{Float64} = Vector{Float64}(undef, 0),
  name::String = "generic-BBModel",
) where {S <: AbstractVector{<:Real}, P, F1 <: Function, F2 <: Function, M <: AbstractNLPModel{P}}
  length(problems) > 0 || error("No problems given")

  nvar = length(x0)
  ncon = length(lcon)
  lvar = convert(S, lvar)
  uvar = convert(S, uvar)
  meta = BBModelMeta(
    nvar,
    x0,
    x_n=x_n,
    lvar = lvar,
    uvar = uvar,
    ncon = ncon,
    lcon = lcon,
    ucon = ucon,
    minimize = true,
    name = name,
  )
  bb_results = bb_results = [Vector{ProblemMetrics}() for _ in 1:length(problems)]
  problems = Dict{Int,Problem{P}}(id => Problem(id, p, eps(P)) for (id, p) ∈ enumerate(problems))
  
  return BBModel(meta, Counters(), solver_function, auxiliary_function, problems, bb_results)
end

# By default, this function will return the time in seconds
function NLPModels.obj(nlp::BBModel{T,S}, x::S) where {T, S}
  problems = nlp.problems
  solver_function = nlp.solver_function
  total_time = 0.0
  try
    for problem in problems
     total_time += @elapsed solver_function(get_nlp(problem), x)
    end
    return total_time
  catch e
    @error "The following error has occured while evaluating the black box: $e"
    return Inf
  end
end

# Function to use for NOMAD: assumes that an interface will sanitize Nomad's output
function obj!(nlp::BBModel{T,S,P}, v::S, p::Problem{P}) where {T, S, P}
  haskey(nlp.problems, get_id(p)) || error("Problem could not be found in problem set")

  solver_function = nlp.solver_function
  auxiliary_function = nlp.auxiliary_function
  nlp_to_solve = get_nlp(p)

  bmark_result, stat = @benchmark_with_result $solver_function($nlp_to_solve, $v) seconds = 10 samples = 5 evals = 1
  times = bmark_result.times
  normalize_times!(times)
  memory = bmark_result.memory
  solved = !is_failure(stat)
  counters = deepcopy(nlp_to_solve.counters)
  p_metric = ProblemMetrics(get_id(p), times, memory, solved, counters)

  reset!(nlp_to_solve)
  update_problem_weight!(nlp, p_metric)
  push!(nlp.bb_results[get_pb_id(p_metric)], p_metric)
  return auxiliary_function(p_metric)
end

function update_problem_weight!(nlp::BBModel, p_metric::ProblemMetrics)
  problems = nlp.problems
  p = problems[get_pb_id(p_metric)]
  p.weight += median(get_times(p_metric))
end

# NLPModels functions to overload:

NLPModels.grad!(::BBModel, ::AbstractVector, ::AbstractVector) = error("Cannot evaluate gradient of a BBModel")

NLPModels.hess_structure!(::BBModel, ::AbstractVector, ::AbstractVector) = error("Cannot get the Hessian of a BBModel")

NLPModels.hess_coord!(::BBModel, ::AbstractVector, ::AbstractVector; obj_weight=1.0) = error("Cannot evaluate the objective Hessian of a BBModel")

NLPModels.hprod!(::BBModel, ::AbstractVector, ::AbstractVector, ::AbstractVector; obj_weight=1) = error("Cannot obtain the  objective Hessian of a BBModel")

function NLPModels.reset_data!(nlp::BBModel)
  problem_collection = (get_nlp(p) for p ∈ nlp.problems)
  for p ∈ problem_collection
    reset!(p)
  end
  empty!(nlp.bb_results)

  return nlp
end
# TODO: overload function from NLPModels for constrained models