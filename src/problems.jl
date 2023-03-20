export Problem,
  get_nlp,
  get_id,
  ProblemMetrics,
  get_pb_id,
  get_times,
  get_memory,
  get_nb_eval,
  get_status,
  get_counters,
  is_failure

export time_only, memory_only, sumfc

"""Mutable struct `Problem`

Keep track of an instance of an `AbstractNLPModel` in a distributed context by giving each instance an `id` and a non-negative `weight`.

The following constructor is available:

    Problem(id::Int, nlp::Union{AbstractNLPModel, Function}, weight::Float64)

Assign an `id` and a `weight` to the problem `nlp`. Note that `nlp` can be given as an `Function` returning an `AbstractNLPModel` once evaluated.
"""
mutable struct Problem{M <: Union{AbstractNLPModel, Function}}
  id::Int
  nlp::M
  weight::Float64
  function Problem(
    id::Int,
    nlp::M,
    weight::Float64 = eps(),
  ) where {M <: Union{AbstractNLPModel, Function}}
    weight ≥ 0 || error("weight of a problem should be greater or equal to 0")
    new{M}(id, nlp, weight)
  end
end

"""Constructor of a `Problem`. Takes an id and an `AbstractNLPModel`."""
Problem(id::Int, nlp::AbstractNLPModel) = Problem(id, nlp, eps(Float64))

"""Returns the `AbstractNLPModel` of a `Problem`."""
get_nlp(p::Problem) = p.nlp
get_nlp(p::Problem{F}) where {F <: Function} = p.nlp()

"""Returns the id of a `Problem`."""
get_id(p::Problem) = p.id

"""
    ProblemMetrics

Structure that contains metrics of a given solver applied to an `AbstractNLPModel`.

The following metrics are stored:

- `pb_id::Int`: identifier of the problem solved, see [`Problem`](@ref);
- `times::Vector{Float64}`: Execution time of the attempts of solving the nlp;
- `memory::Int`: Memory allocated to solve the nlp;
- `status::Symbol`: Status of the nlp after solve, see [SolverCore.jl's documentation](https://juliasmoothoptimizers.github.io/SolverCore.jl/dev/);
- `counters::Counters`: Counters of the nlp, see [NLPModels.jl's documentation](https://juliasmoothoptimizers.github.io/NLPModels.jl/dev/tools/).
"""
struct ProblemMetrics
  pb_id::Int
  times::Vector{Float64}
  memory::Int
  status::Symbol
  counters::Counters

  function ProblemMetrics(
    id::Int64,
    times::Vector{Float64},
    memory::Int64,
    status::Symbol,
    counters::Counters,
  )
    new(id, times, memory, status, counters)
  end
end

ProblemMetrics(id::Int64, t::Tuple{Vector{Float64}, Int64, Symbol, Counters}) =
  ProblemMetrics(id, t...)

"""Returns the id of the problem linked to this `ProblemMetrics` instance."""
get_pb_id(p::ProblemMetrics) = p.pb_id

"""Returns the times required in seconds to solve the problem linked to this `ProblemMetrics` instance."""
get_times(p::ProblemMetrics) = p.times ./ 1.0e9

"""Returns the memory allocated in Mb to solve the problem linked to this `ProblemMetrics` instance."""
get_memory(p::ProblemMetrics) = p.memory ÷ (2^20)

"""Returns the problem linked to this `ProblemMetrics` instance is solved."""
get_status(p::ProblemMetrics) = p.status

"""Returns the Counters related the problem linked to this `ProblemMetrics` instance."""
get_counters(p::ProblemMetrics) = p.counters

"""
    time_only(vec_metric::Vector{ProblemMetrics}; penalty::Float64 = 5.0)

Return the median time, if more than one solve, of `p_metric`.
Unsolved problems are penalyzed by a `penalty` factor.
"""
function time_only(vec_metric::Vector{ProblemMetrics}; penalty::Float64 = 5.0)
  total = 0.0
  for p_metric in vec_metric
    total += median(get_times(p_metric)) + is_failure(get_status(p_metric)) * penalty
  end
  return total
end

"""
    memory_only(vec_metric::Vector{ProblemMetrics}; penalty::Float64 = 5.0)

Return the memory used in `p_metric`.
Unsolved problems are penalyzed by a `penalty` factor.
"""
function memory_only(vec_metric::Vector{ProblemMetrics}; penalty::Float64 = 5.0)
  total = 0.0
  for p_metric in vec_metric
    total += get_memory(p_metric) + is_failure(get_status(p_metric)) * penalty
  end
  return total
end

"""
    sumfc(vec_metric::Vector{ProblemMetrics}; penalty::Float64 = 5.0)

Return the sum of the evaluations of the objective function and constraint function.
Unsolved problems are penalyzed by a `penalty` factor.
"""
function sumfc(vec_metric::Vector{ProblemMetrics}; penalty::Float64 = 5.0)
  total = 0.0
  for p_metric in vec_metric
    counters = get_counters(p_metric)
    total += counters.neval_obj + counters.neval_cons + is_failure(get_status(p_metric)) * penalty
  end
  return total
end
