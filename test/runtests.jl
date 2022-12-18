using BBModels
using SolverCore
using NLPModels
using ADNLPModels
using OptimizationProblems, OptimizationProblems.ADNLPProblems
using JSOSolvers
using Test
using SolverParameters

T = Float64
n = 5
problems = (
  eval(p)(type = Val(T)) for
  (_, p) ∈ zip(1:n, filter(x -> x != :ADNLPProblems, names(OptimizationProblems.ADNLPProblems)))
)
problems =
  Iterators.filter(p -> unconstrained(p) && 1 ≤ get_nvar(p) ≤ 100 && get_minimize(p), problems)
problems = collect(problems)

@testset "BBModels.jl" verbose = true begin
  include("param_structs.jl")
  include("test_utils.jl")
  include("benchmark_macros_test.jl")
  include("bbmodels_test.jl")
  include("problems_test.jl")
end
