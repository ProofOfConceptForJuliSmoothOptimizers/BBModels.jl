@testset "utils function" verbose = true begin
  @testset "test `normalize_times!`:" begin
    original_times = [rand() * 1e6]
    normalized_times = copy(original_times)
    cpu_frequency = BBModels.get_cpu_frequency()
    BBModels.normalize_times!(normalized_times)
    @test (normalized_times .* cpu_frequency) ≈ original_times
  end

  @testset "test `is_failure` function" verbose = true begin
    nlp = ADNLPModel(x -> dot(x, x), zeros(2))
    stats = GenericExecutionStats(nlp)
    set_status!(stats, :first_order)
    @test BBModels.is_failure(stats) == false
    @testset "Test failure status: $failure_status" verbose = true for failure_status in [
      :exception,
      :infeasible,
      :max_eval,
      :max_iter,
      :max_time,
      :stalled,
      :neg_pred,
    ]
      stats = GenericExecutionStats(nlp)
      set_status!(stats, failure_status)
      @test BBModels.is_failure(stats) == true
    end
  end
end
