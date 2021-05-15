#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PASS_CONTEXT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PASS_CONTEXT_H_

#include <string>
#include <utility>

#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// A global context to pass arguments from python to xla passes
namespace pass_context {

// TODO: replace this with absl::any in "absl/types/any.h"
class AnyObject {
 public:
  enum class Type : uint8 {
    kDouble,
    kInt,
    kString,
    kIntVector,
    kDoubleVector,
    kNone,
  };

  Type type;
  double double_val;
  int64 int_val;
  std::string str_val;
  std::vector<int64> int_vector_val;
  std::vector<double> double_vector_val;
};

// Read context values from a pyton dict
void SetPassContext(pybind11::dict dict);

// Clear context values
void ClearPassContext();

int64 GetInt(const std::string& name, int64 default_value);

bool GetBool(const std::string& name, bool default_value);

double GetDouble(const std::string& name);

std::string GetString(const std::string& name, const std::string& default_value);

std::vector<int64> GetIntVector(const std::string& name);

std::vector<double> GetDoubleVector(const std::string& name);

}  // namespace pass_context
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_PASS_CONTEXT_H_
