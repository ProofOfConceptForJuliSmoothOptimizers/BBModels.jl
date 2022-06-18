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

mutable struct Problem
  id::Int
  nlp::AbstractNLPModel
  weight::Float64
  function Problem(id::Int, nlp::AbstractNLPModel, weight::Float64) 
    weight ≥ 0 || error("weight of a problem should be greater or equal to 0")
    new(id, nlp, weight)
  end
end

Problem(id::Int, nlp::AbstractNLPModel) = Problem(id, nlp, eps(Float64))

get_nlp(p::Problem)  = p.nlp
get_id(p::Problem) = p.id

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
    times ./= 1.0e9
    memory = round(Int, memory / 1.0e6)
    new(id, times, memory, solved, counters)
  end
end

ProblemMetrics(id::Int64, t::Tuple{Vector{Float64}, Int64, Bool, Counters}) =
  ProblemMetrics(id, t...)

get_pb_id(p::ProblemMetrics) = p.pb_id
get_times(p::ProblemMetrics) = p.times
get_memory(p::ProblemMetrics) = p.memory
get_solved(p::ProblemMetrics) = p.solved
get_counters(p::ProblemMetrics) = p.counters
