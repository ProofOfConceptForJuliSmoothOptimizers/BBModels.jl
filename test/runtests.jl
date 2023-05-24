using BBModels
using SolverCore
using NLPModels
using ADNLPModels
using OptimizationProblems, OptimizationProblems.ADNLPProblems
using Test
using SolverParameters
using Statistics
using LinearAlgebra

T = Float64
n = 5

meta = OptimizationProblems.meta
list = meta[meta.minimize .& (meta.ncon .== 0) .& .!meta.has_bounds .& (meta.nvar .≤ 100), :name]
list = intersect(Symbol.(list), names(OptimizationProblems.ADNLPProblems)) # optional
problems = [eval(p)(type = Val(T)) for (_, p) ∈ zip(1:n, list)]
problems_expr = [() -> eval(p)(type = Val(T)) for (_, p) ∈ zip(1:n, list)]

@testset "BBModels.jl" verbose = true begin
  include("test_random_search.jl")
  include("param_structs.jl")
  include("test_utils.jl")
  include("benchmark_macros_test.jl")
  include("bbmodels_test.jl")
  include("problems_test.jl")
end
