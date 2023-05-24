using SolverParameters

struct LBFGSParameterSet{T <: Real} <: AbstractParameterSet
  mem::Parameter{Int, IntegerRange{Int}}
  τ₁::Parameter{T, RealInterval{T}}
  bk_max::Parameter{Int, IntegerRange{Int}}
  # add scaling
  
  function LBFGSParameterSet{T}(;mem::Int = 5, τ₁::T = T(0.9999), bk_max::Int = 25) where {T}        
    p_set = new(
      Parameter(mem, IntegerRange(Int(1), Int(20)), "mem"),
      Parameter(τ₁, RealInterval(T(0), T(1)), "τ₁"),
      Parameter(bk_max, IntegerRange(Int(10), Int(50)), "bk_max"),
    )
    return p_set
  end

  function LBFGSParameterSet(;kwargs...)
    return LBFGSParameterSet{Float64}(; kwargs...)
  end
end

include("lbfgs.jl")

param_set = LBFGSParameterSet()
subset = (:mem, ) # optimize only `mem`

using BenchmarkTools

function fun(vec_metrics::Vector{ProblemMetrics})
  penalty = 1e2
  global fx = 0
  for p in vec_metrics
    failed = is_failure(BBModels.get_status(p))
    nobj = get_counters(p).neval_obj
    med_time = BenchmarkTools.median(get_times(p))
    fx += failed * penalty + nobj + med_time
  end
  return fx
end

model = BBModel(
  param_set, # AbstractParameterSet
  problems, # vector of AbstractNLPModel
  lbfgs, # (::AbstractNLPModel, ::AbstractParameterSet) -> GenericExecutionStats
  fun, # time_only, memory_only, sumfc OR a hand-made function
  subset = subset,
)

vals = BBModels.random_search(model, verbose = 0)
set_values!(subset, param_set, vals)

model = BBModel(
  param_set, # AbstractParameterSet
  problems, # vector of AbstractNLPModel
  lbfgs, # (::AbstractNLPModel, ::AbstractParameterSet) -> GenericExecutionStats
  fun, # time_only, memory_only, sumfc OR a hand-made function
)
vals = BBModels.random_search(model, verbose = 0)
set_values!(param_set, vals)
