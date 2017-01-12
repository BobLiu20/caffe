#include <algorithm>
#include <vector>

#include "caffe/layers/blackhole_layer.hpp"

#include "caffe/util/math_functions.hpp"
namespace caffe {

#ifdef CPU_ONLY
  STUB_GPU(BlackHoleLayer);
#endif

  INSTANTIATE_CLASS(BlackHoleLayer);
  REGISTER_LAYER_CLASS(BlackHole);
}
