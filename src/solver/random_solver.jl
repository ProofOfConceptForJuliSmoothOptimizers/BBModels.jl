"""
    random_search(model::BBModel; verbose = 0, max_time = 30.0, max_iter = 10)

Generate random values of the parameter set.

Keywords arguments:
- `check_cache :: Bool = true`: Check if `x` has already been evaluated;
- `verbose :: Integer = 0`: Print iteration information if `>0`;
- `max_time :: Float64 = 30.0 `: time limit in seconds;
- `max_iter :: Integer = 10`: maximum number of iterations.
"""
function random_search(
  model::BBModel;
  check_cache = true,
  verbose = 0,
  max_time = 30.0,
  max_iter = 10,
)
  cache = Float64[]
  cache_x = Any[]
  start_time = time()
  for i = 1:max_iter
    time() - start_time > max_time && break
    x = SolverParameters.rand(model.subset, model.parameter_set) # may return categorical variables
    if !check_cache || !(x in cache_x)
      fx = BBModels.obj_cat(model, x)
      push!(cache, fx)
      push!(cache_x, x)
      (verbose > 0) && println("$i: fx=$fx")
    end
  end
  is = argmin(cache)
  println("Best value give mem=$(cache[is])")
  return cache_x[is]
end
