#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <stdlib.h>
#include <stdio.h>



namespace spalpa {

template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) && std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From& src) noexcept {
  static_assert(
      std::is_trivially_constructible<To>::value,
      "This implementation additionally requires destination type to be trivially constructible");

  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

// Packs a descriptor object into a byte string.
template <typename T>
std::string PackDescriptorAsString(const T& descriptor) {
  return std::string(bit_cast<const char*>(&descriptor), sizeof(T));
}

// Unpacks a descriptor object from a byte string.
template <typename T>
const T* UnpackDescriptor(const char* opaque, std::size_t opaque_len) {
  if (opaque_len != sizeof(T)) {
    throw std::runtime_error("Invalid opaque object size");
  }
  return bit_cast<const T*>(opaque);
}

struct DnMatDescr_t {
  int rows;
  int cols;
} typedef DnMatDescr_t;

DnMatDescr_t build_DnMatDescr(int rows, int cols) {
    return DnMatDescr_t{rows, cols};
}

struct SparseShardingDescr_t {
  char sharding;
} typedef SparseShardingDescr_t;

SparseShardingDescr_t build_SparseShardingDescr(char sharding) {
    return SparseShardingDescr_t{sharding};
}

struct SpmmDescr_t {
  DnMatDescr_t B;
  SparseShardingDescr_t sparse_sharding;
} typedef SpmmDescr_t;

SpmmDescr_t build_SpmmDescr(DnMatDescr_t B, SparseShardingDescr_t sparse_sharding) {
    return SpmmDescr_t{B, sparse_sharding};
}


}