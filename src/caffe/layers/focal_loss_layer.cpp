#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void FocalLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  alpha_ = this->layer_param_.focal_loss_param().alpha();
  gamma_ = this->layer_param_.focal_loss_param().gamma();
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);

  const Dtype* prob_data = sigmoid_output_->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
//  const Dtype* input_data = bottom[0]->cpu_data();
  Dtype loss = 0;
  Dtype pt = 0;
  Dtype alphat;
  int inner_num_ = bottom[0]->count();
  int count_pos = 0;
  int count_neg = 0;

  for (int j = 0; j < inner_num_; j++) {
    const int label_value = static_cast<int>(target[j]);
	if (label_value == 1) count_pos++; else count_neg++;
  }
  alpha_ = 1.0*count_neg/(count_pos+count_neg);

  for (int j = 0; j < inner_num_; j++) {
    const int label_value = static_cast<int>(target[j]);
    DCHECK_GE(label_value, 0);
    pt = (label_value == 1) ? prob_data[j] : (1-prob_data[j]);
	alphat = (label_value == 1) ? alpha_ : (1-alpha_);
    loss -= alphat * pow(1.0 - pt, gamma_) * log(std::max(pt, Dtype(FLT_MIN)));
  }
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype focal_diff = 0;
    Dtype pt = 0;
    Dtype pc = 0;
	Dtype alphat;
	int count_neg = 0;
	int count_pos = 0;

	/* compute alpha_ */
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(target[j]);
      if (label_value == 1) count_pos++; else count_neg++;
    }
    alpha_ = 1.0*count_neg/(count_pos+count_neg);

    for (int j = 0; j < bottom[0]->count(); ++j) {
      const int label_value = static_cast<int>(target[j]);
	  if (label_value == 1) {
		pt = prob_data[j];
		alphat = alpha_;
		focal_diff = alphat * pow(1 - pt, gamma_) * (gamma_ * pt * log(std::max(pt, Dtype(FLT_MIN))) + pt - 1);
	  } else {
		pc = prob_data[j];
		alphat = 1 - alpha_;
		focal_diff = -1.0 * alphat * pow(pc, gamma_) * (gamma_ * (1-pc) * log(std::max(1-pc, Dtype(FLT_MIN))) - pc);
	  }
      bottom_diff[j] = focal_diff;
	}
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(bottom[0]->count(),loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(FocalLossLayer);
#endif

INSTANTIATE_CLASS(FocalLossLayer);
REGISTER_LAYER_CLASS(FocalLoss);

}  // namespace caffe
