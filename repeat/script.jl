using Pkg
Pkg.activate(".")
# Pkg.add(url="https://github.com/MonssafToukal/SolverParameters.jl", rev="main")
Pkg.add(url="https://github.com/tmigot/SolverParameters.jl", rev="fix-rand")
Pkg.add(url="https://github.com/ProofOfConceptForJuliSmoothOptimizers/BBModels.jl", rev="main")

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

using BBModels
using ADNLPModels, OptimizationProblems, OptimizationProblems.ADNLPProblems

n = 5
meta = OptimizationProblems.meta
list = meta[
  meta.minimize .& (meta.ncon .== 0) .& .!meta.has_bounds .& (20 .≤ meta.nvar .≤ 100),
  :name
]
problems = [eval(p)() for p ∈ Symbol.(list[1:n])]

nvars = [p.meta.nvar for p in problems]
@show nvars

param_set = LBFGSParameterSet()
solver_func = lbfgs
subset = (:mem, ) # optimize only `mem`

using BenchmarkTools

function fun(vec_metrics::Vector{ProblemMetrics})
  penalty = 1e2
  global fx = 0
  for p in vec_metrics
    failed = is_failure(BBModels.get_status(p))
    nobj = get_counters(p).neval_obj
    med_time = BenchmarkTools.median(get_times(p))
    @info "Problem $(p.pb_id) - $(problems[p.pb_id].meta.name) : $(failed) #f:$(nobj) t=$(med_time)"
    fx += failed * penalty + nobj + med_time
  end
  return fx
end

model = BBModel(
  param_set, # AbstractParameterSet
  problems, # vector of AbstractNLPModel
  solver_func, # (::AbstractNLPModel, ::AbstractParameterSet) -> GenericExecutionStats
  fun, # time_only, memory_only, sumfc OR a hand-made function
  subset = subset,
)

function very_smart_algo(model::BBModel; verbose = 0, max_time = 30.0, max_iter = 10)
  cache = Float64[]
  cache_x = Any[]
  start_time = time()
  for i=1:max_iter
    time() - start_time > max_time && break
    x = SolverParameters.rand(model.subset, model.parameter_set) # may return categorical variables
    @show x
    fx = BBModels.obj_cat(model, x)
    push!(cache, fx)
    push!(cache_x, x)
    (verbose > 0) && println("$i: fx=$fx")
  end
  is = argmin(cache)
  println("Best value is mem=$(cache_x[is])")
  return cache_x[is]
end

vals = very_smart_algo(model, verbose = 1)
set_values!(subset, param_set, vals)
