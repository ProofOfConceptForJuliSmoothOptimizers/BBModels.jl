function solver_func(nlp::AbstractNLPModel, p::NamedTuple)
  return lbfgs(nlp; verbose=0, max_time=60.0)
end

function aux_func(p_metric::ProblemMetrics)
  median_time = median(get_times(p_metric))
  memory = get_memory(p_metric)
  solved = get_solved(p_metric)
  counters = get_counters(p_metric)
  return median_time + memory + counters.neval_obj + (Float64(!solved) * 5.0 * median_time)
end

@testset "Testing BBModels core functions" verbose=true begin
  x = (mem=5, scaling=true, τ₁=T(0.999), bk_max=20)
  nlp = BBModel(x, solver_func, aux_func, problems)
  lvar = Real[1, false, T(0.0), 10]
  uvar = Real[100, true, T(0.9999), 30]
  
  @test eltype(nlp.meta.x0) == Union{(typeof(xᵢ) for xᵢ in x)...}

  nlp = BBModel(x, solver_func, aux_func, problems;lvar=lvar, uvar=uvar)

  @test [typeof(l) for l in nlp.meta.lvar] == [Int, Bool, T, Int]
  @test [typeof(l) for l in nlp.meta.uvar] == [Int, Bool, T, Int]

end