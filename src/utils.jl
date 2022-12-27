"""Returns `true` if a status is considered a failure."""
function is_failure(stats::AbstractExecutionStats)
  failure_status = [:exception, :infeasible, :max_eval, :max_iter, :max_time, :stalled, :neg_pred]
  return any(s -> s == stats.status, failure_status)
end

"""Modifies in place a Vector of times by dividing each value by the frequency of the CPU in Mz."""
function normalize_times!(times::Vector{Float64})
  times ./= get_cpu_frequency()
end

"""Returns the CPU frequency in GHz of the host executing this function."""
function get_cpu_frequency()
  return first(Sys.cpu_info()).speed / 1.0e3
end
