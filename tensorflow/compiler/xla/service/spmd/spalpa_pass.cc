#include "tensorflow/compiler/xla/service/spalpa_pass.h"

#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "tensorflow/compiler/xla/service/heap_simulator.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/service/spmd/auto_sharding_strategy.h"
#include "tensorflow/compiler/xla/service/hlo


namespace xla {
namespace spmd {


StatusOr<bool> HandleSparsePartition::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {

//   Array<int64_t> device_mesh(
//       pass_context::GetIntVector("auto_sharding::device_mesh_shape"));
//   device_mesh.SetValues(
//       pass_context::GetIntVector("auto_sharding::device_mesh_ids"));
//   ProfilingResult prof_result(
//       pass_context::GetPyObject("auto_sharding::device_mesh_prof_result"));
//   ClusterEnvironment cluster_env(
//       device_mesh,
//       pass_context::GetDoubleVector("auto_sharding::device_mesh_alpha"),
//       pass_context::GetDoubleVector("auto_sharding::device_mesh_beta"),
//       prof_result, solver_option);

  // std::cerr << "===== Enter AutoSharding =====" << std::endl;
  // std::cerr << module->ToString();
  // std::cerr << "=====================================" << std::endl;

  const HloComputation* entry_computation = module->entry_computation();
  std::unique_ptr<HloAliasAnalysis> alias_analysis =
      HloAliasAnalysis::Run(module).ValueOrDie();
  AliasMap alias_map =
      BuildAliasMap(module, alias_analysis->dataflow_analysis());

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloLiveRange> hlo_live_range,
      HloLiveRange::Run(schedule, *alias_analysis, entry_computation));
  absl::flat_hash_map<const HloValue*, HloLiveRange::TimeBound>&
      buffer_live_ranges = hlo_live_range->buffer_live_ranges();
  LivenessSet liveness_set(hlo_live_range->schedule_end_time() + 1);
  for (const auto& iter : buffer_live_ranges) {
    for (int64_t i = iter.second.start; i <= iter.second.end; ++i) {
      liveness_set[i].push_back(iter.first);
    }
  }

  const HloInstructionSequence& sequence =
      hlo_live_range->flattened_instruction_sequence();
  std::vector<const HloInstruction*> instructions(sequence.instructions().begin(),
                                                 sequence.instructions().end());
}

}  // namespace spmd
}  // namespace xla
