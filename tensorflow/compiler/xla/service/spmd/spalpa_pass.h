#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_SPALPA_PASS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_SPALPA_PASS_H_

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace spmd {

class HandleSparsePartition : public HloModulePass {
 public:
  HandleSparsePartition() = default;
  ~HandleSparsePartition() override = default;
  absl::string_view name() const override { return "HandleSparsePartition"; }

  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace spmd
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_SPALPA_PASS_H_
