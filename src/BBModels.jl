module BBModels

using Distributed
# TODO: keep this until the SolverParameters is registered
using Statistics
using StaticArrays

using LinearOperators, NLPModels, NLPModelsModifiers, SolverCore

using BenchmarkTools
using BenchmarkTools: Trial
import BenchmarkTools.hasevals
import BenchmarkTools.prunekwargs
import BenchmarkTools.Benchmark
import BenchmarkTools.Parameters
import BenchmarkTools.run_result

using SolverParameters

include("utils.jl")
include("benchmark_macros.jl")
include("problems.jl")
include("meta.jl")
include("bb_models.jl")

const PROBLEM_PARTITION = Vector{ProblemMetrics}()

end
