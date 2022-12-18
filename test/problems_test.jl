@testset "Test Problems struct functions" verbose = true begin
  p1 = Problem(1, problems[1])

  @test get_nlp(p1) === problems[1]
  @test get_id(p1) == 1

  bmark, stats = BBModels.@benchmark_with_result lbfgs($problems[1]; mem = 15)
  BBModels.normalize_times!(bmark.times)
  p_metrics = ProblemMetrics(
    1,
    bmark.times,
    bmark.memory,
    !BBModels.is_failure(stats),
    deepcopy(problems[1].counters),
  )

  @test get_pb_id(p_metrics) == 1
  @test get_times(p_metrics) === bmark.times
  @test get_memory(p_metrics) == bmark.memory
  @test get_solved(p_metrics) == !BBModels.is_failure(stats)
  for field in fieldnames(Counters)
    @test getfield(get_counters(p_metrics), field) == getfield(problems[1].counters, field)
  end
  @test get_counters(p_metrics) !== problems[1].counters
  reset!(problems[1])
  for field in fieldnames(Counters)
    @test getfield(get_counters(p_metrics), field) >= getfield(problems[1].counters, field)
  end
end
