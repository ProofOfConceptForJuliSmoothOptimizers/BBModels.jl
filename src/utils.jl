"""
    is_failure(status::Symbol)
    is_failure(stats::AbstractExecutionStats)

Returns `true` if a status is considered a failure.
`SolverCore.show_statuses()` return the list of all possible statuses.
"""
function is_failure(status::Symbol)
  failure_status = [
    :exception,
    :infeasible,
    :max_eval,
    :max_iter,
    :max_time,
    :stalled,
    :neg_pred,
    :not_desc,
    :small_step,
    :unknown,
    :user,
  ]
  return any(s -> s == status, failure_status)
end

is_failure(stats::AbstractExecutionStats) = is_failure(stats.status)

"""Modifies in place a Vector of times by dividing each value by the frequency of the CPU in Mz."""
function normalize_times!(times::Vector{Float64})
  times ./= get_cpu_frequency()
end

"""Returns the CPU frequency in GHz of the host executing this function."""
function get_cpu_frequency()
  return first(Sys.cpu_info()).speed / 1.0e3
end
