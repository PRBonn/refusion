// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de)
#include "tsdfvh/heap.h"

namespace refusion {

namespace tsdfvh {

void Heap::Init(int heap_size) {
  cudaMallocManaged(&heap_, sizeof(unsigned int) * heap_size);
}

__device__ unsigned int Heap::Consume() {
  unsigned int idx = atomicSub(&heap_counter_, 1);
  return heap_[idx];
}

__device__ void Heap::Append(unsigned int ptr) {
  unsigned int idx = atomicAdd(&heap_counter_, 1);
  heap_[idx + 1] = ptr;
}


}  // namespace tsdfvh

}  // namespace refusion
