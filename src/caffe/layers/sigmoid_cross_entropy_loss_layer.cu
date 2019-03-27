#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
__global__ void SigmoidCrossEntropyLossForwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* target, Dtype* loss) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const int target_value = static_cast<int>(target[i]);
	if (target_value == 1) { /* positive */
      loss[i] = input_data[i] * (1 - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] *
        (input_data[i] >= 0)));
	} else { /* negative */
      loss[i] = input_data[i] * (0 - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] *
        (input_data[i] >= 0)));
	}
  }
}

template <typename Dtype>
__global__ void SigmoidCrossEntropyLossBackwardGPU(const int nthreads,
          Dtype* diff, const Dtype* target, double weight_pos, double weight_neg) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const int target_value = static_cast<int>(target[i]);
	if (target_value == 1) { /* positive */
      diff[i] *= weight_pos;
	} else { /* negative */
      diff[i] *= weight_neg;
	}
  }
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);

  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* target = bottom[1]->gpu_data();

  const int count = bottom[0]->count();
  const int dim = count / bottom[0]->channels();
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();

  SigmoidCrossEntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, input_data, target, loss_data);

  Dtype count_pos  = 0;
  Dtype count_neg  = 0;
  double loss_pos  = 0;
  double loss_neg  = 0;
  const Dtype* target_cpu = bottom[1]->cpu_data();
  /* calculate beta */
  for (int i = 0; i < dim; i++) {
	if (target_cpu[i] == 1) count_pos++;
	else count_neg++;
  }
  weight_pos_ = 1.0 * count_neg / (count_pos + count_neg);
  weight_neg_ = 1.0 * count_pos / (count_pos + count_neg);

  /* calculate loss for positive and negative pixels */
  const Dtype* loss_data_cpu = bottom[0]->cpu_diff();
  for (int i = 0; i < dim; i++) {
	if (target_cpu[i] == 1) 
	  loss_pos -= (double)loss_data_cpu[i];
	else
	  loss_neg -= (double)loss_data_cpu[i];
  }
  loss_pos *= weight_pos_;
  loss_neg *= weight_neg_;

  top[0]->mutable_cpu_data()[0] = (loss_pos * 1 + loss_neg);
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_sub(count, sigmoid_output_data, target, bottom_diff);
    int dim = bottom[0]->count() / bottom[0]->num();

    SigmoidCrossEntropyLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_diff, target, (double)weight_pos_, (double)weight_neg_);

    const Dtype loss_weight = top [0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidCrossEntropyLossLayer);

}  // namespace caffe
