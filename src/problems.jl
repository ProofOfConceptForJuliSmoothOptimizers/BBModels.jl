export Problem,
  get_nlp,
  get_id,
  ProblemMetrics,
  get_pb_id,
  get_times,
  get_memory,
  get_nb_eval,
  get_solved,
  get_counters

export time_only, memory_only, sumfc

"""Mutable struct encapsulating a NLPModel.
The goal is to keep track of an instance of an `AbstractNLPModel` in a distributed context by giving each instance an `id` and a `weight`.
"""
mutable struct Problem
  id::Int
  nlp::AbstractNLPModel
  weight::Float64
  function Problem(id::Int, nlp::AbstractNLPModel, weight::Float64)
    weight ≥ 0 || error("weight of a problem should be greater or equal to 0")
    new(id, nlp, weight)
  end
end

"""Constructor of a `Problem`. Takes an id and an `AbstractNLPModel`."""
Problem(id::Int, nlp::AbstractNLPModel) = Problem(id, nlp, eps(Float64))

"""Returns the `AbstractNLPModel` of a `Problem`."""
get_nlp(p::Problem) = p.nlp

"""Returns the id of a `Problem`."""
get_id(p::Problem) = p.id

"""Struct that contains metrics of a given `AbstractNLPModel`.
The goal is to measure the performance of an arbitrary solver solving a particular `AbstractNLPModel`.
The following metrics are stored:

1. times → Execution time of one or many attempts of solving the nlp.
2. memory → Memory allocated to solve the nlp.
3. solved → Status of the nlp after solve. this attribute is `true` if nlp is solved and `false` otherwise.
4. Counters → Struct containing counters to the evaluations of certain methods related to the nlp (e.g, number of evaluations of the objective).
"""
struct ProblemMetrics
  pb_id::Int
  times::Vector{Float64}
  memory::Int
  solved::Bool
  counters::Counters

  function ProblemMetrics(
    id::Int64,
    times::Vector{Float64},
    memory::Int64,
    solved::Bool,
    counters::Counters,
  )
    new(id, times, memory, solved, counters)
  end
end

ProblemMetrics(id::Int64, t::Tuple{Vector{Float64}, Int64, Bool, Counters}) =
  ProblemMetrics(id, t...)

"""Returns the id of the problem linked to this `ProblemMetrics` instance."""
get_pb_id(p::ProblemMetrics) = p.pb_id

"""Returns the times required to solve the problem linked to this `ProblemMetrics` instance."""
get_times(p::ProblemMetrics) = p.times

"""Returns the memory allocated to solve the problem linked to this `ProblemMetrics` instance."""
get_memory(p::ProblemMetrics) = p.memory

"""Returns the problem linked to this `ProblemMetrics` instance is solved."""
get_solved(p::ProblemMetrics) = p.solved

"""Returns the Counters related the problem linked to this `ProblemMetrics` instance."""
get_counters(p::ProblemMetrics) = p.counters

"""
    time_only(p_metric::ProblemMetrics; penalty::Float64 = 5.0)

Return the median time, if more than one solve, of `p_metric`.
Unsolved problems are penalyzed by a `penalty` factor.
"""
function time_only(p_metric::ProblemMetrics; penalty::Float64 = 5.0)
  median(get_times(p_metric)) + !(get_solved(p_metric)) * penalty
end

"""
    memory_only(p_metric::ProblemMetrics; penalty::Float64 = 5.0)

Return the memory used in `p_metric`.
Unsolved problems are penalyzed by a `penalty` factor.
"""
memory_only(p_metric::ProblemMetrics; penalty::Float64 = 5.0) =
  get_memory(p_metric) + !(get_solved(p_metric)) * penalty

"""
    sumfc(p_metric::ProblemMetrics; penalty::Float64 = 5.0)

Return the sum of the evaluations of the objective function and constraint function.
Unsolved problems are penalyzed by a `penalty` factor.
"""
function sumfc(p_metric::ProblemMetrics; penalty::Float64 = 5.0)
  counters = get_counters(p_metric)
  return counters.neval_obj + counters.neval_cons + !(get_solved(p_metric)) * penalty
end
