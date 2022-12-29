# https://discourse.julialang.org/t/retain-computation-results-with-btime/62644/8
macro benchmark_with_result(args...)
  _, params = prunekwargs(args...)
  bench = gensym()
  tune_phase = hasevals(params) ? :() : :($BenchmarkTools.tune!($bench))
  return esc(quote
    local $bench = $BenchmarkTools.@benchmarkable $(args...)
    $BenchmarkTools.warmup($bench)
    $tune_phase
    $BenchmarkTools.run_result($bench)
  end)
end
