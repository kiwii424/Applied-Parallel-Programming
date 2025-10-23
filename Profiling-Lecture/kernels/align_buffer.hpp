#pragma once

#include <cassert>
#include <type_traits>
#include <vector_types.h>

#define DEVICE_ONLY __device__ __forceinline__
#define HOST_DEVICE __host__ __device__ __forceinline__

template <
  typename T,
  size_t N,
  size_t Align = 16
>
struct AlignedSharedMemBuffer {
public:
  /// size of one vector
  static constexpr size_t kVectorSize = Align / sizeof(T);

  /// Number of logical elements held in buffer
  static constexpr size_t kCount = (N + kVectorSize - 1) / kVectorSize * kVectorSize;

private:

  /// Internal storage
  alignas(Align) T storage[kCount];

public:

  //
  // C++ standard members
  //

  typedef T value_type;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef value_type *pointer;
  typedef value_type const * const_pointer;

public:

  HOST_DEVICE
  pointer data() {
    return storage; 
  }

  HOST_DEVICE
  const_pointer data() const {
    return storage; 
  }
  
  HOST_DEVICE 
  value_type&
  operator[](size_type index) {
    // no safety check for performance
    return storage[index];
  }

  HOST_DEVICE
  const value_type&
  operator[](size_type index) const {
    // no safety check for performance
    return storage[index];
  }
  
  // Only allow access as an element or as a vector of elements
  template<typename VectorType = value_type>
  HOST_DEVICE
  std::enable_if_t<std::is_same_v<VectorType, value_type> || sizeof(VectorType) == Align, VectorType &>
  access_vector(size_type vec_index) {
    // no safety check for performance
    return *reinterpret_cast<VectorType *>(storage + vec_index * kVectorSize);
  }

  template<typename VectorType = value_type>
  HOST_DEVICE
  std::enable_if_t<std::is_same_v<VectorType, value_type> || sizeof(VectorType) == Align, const VectorType &>
  access_vector(size_type vec_index) const {
    // no safety check for performance
    return *reinterpret_cast<VectorType *>(storage + vec_index * kVectorSize);
  }

  HOST_DEVICE
  constexpr bool empty() const {
    return !kCount;
  }

  HOST_DEVICE
  constexpr size_type size() const {
    return kCount;
  }

  HOST_DEVICE
  constexpr size_type max_size() const {
    return kCount;
  }
};
