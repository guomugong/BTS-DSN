#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void BinLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  const BinLossParameter  bin_loss_param = this->layer_param_.bin_loss_param();
  lambda_ = bin_loss_param.lambda();
  key_  = bin_loss_param.key();

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
}

template <typename Dtype>
void BinLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[0]->shape(0);  // batch size
  inner_num_ = bottom[0]->count(1);  // instance size: |output| == |target|
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}


template <typename Dtype>
void BinLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  //liuteng - dedao gailv
  const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
  //
  Dtype loss_pos = 0;
  Dtype loss_neg = 0;
  Dtype temp_loss_pos = 0;
  Dtype temp_loss_neg = 0;
  Dtype count_pos = 0;
  Dtype count_neg = 0;
  //liuteng - mei duan xiang su dian shu -neg
  Dtype count_neg1 = 0;
  Dtype count_neg2 = 0;
  Dtype count_neg3 = 0;
  Dtype count_neg4 = 0;
  Dtype count_neg5 = 0;
  Dtype temp_loss_neg1 = 0;
  Dtype temp_loss_neg2 = 0;
  Dtype temp_loss_neg3 = 0;
  Dtype temp_loss_neg4 = 0;
  Dtype temp_loss_neg5 = 0;
  Dtype temp_count_neg = 0;
  //
  Dtype summ = 0;
  //Dtype lambda = 100;


  const int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();

  int valid_count = 0;
  //liuteng
  int valid_count0 = 0;
  int valid_count1 = 0;
  int valid_count2 = 0;
  int valid_count3 = 0;
  int valid_count4 = 0;
  int valid_count5 = 0;
  //

  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
      temp_loss_pos = 0;
      temp_loss_neg = 0;
      count_pos = 0;
      count_neg = 0;
      //liuteng - mei duan xiang su dian shu
      count_neg1 = 0;
      count_neg2 = 0;
      count_neg3 = 0;
      count_neg4 = 0;
      count_neg5 = 0;
      temp_loss_neg1 = 0;
      temp_loss_neg2 = 0;
      temp_loss_neg3 = 0;
      temp_loss_neg4 = 0;
      temp_loss_neg5 = 0;
      temp_count_neg = 0;
      //
      for (int j = 0; j < dim; j ++) {
	const int target_value = static_cast<int>(target[i*dim+j]);
	if (has_ignore_label_ && target_value == ignore_label_) {
	      continue;
	    }
         if (target[i*dim+j] == 1) {
		valid_count0++;
        	count_pos ++;
        	temp_loss_pos -= input_data[i*dim + j] * (target[i*dim+j] - (input_data[i*dim + j] >= 0)) -
                	log(1 + exp(input_data[i*dim + j] - 2 * input_data[i*dim + j] * (input_data[i*dim + j] >= 0)));
    	}
    	else if (target[i*dim+j] == 0) {
        	//liuteng - fen duan ji suan sun shi
		if(sigmoid_output_data[i*dim+j] >= 0 && sigmoid_output_data[i*dim+j] < 0.2){
			
			valid_count1++;
			count_neg1++;
			temp_loss_neg1 -= input_data[i*dim + j] * (target[i*dim+j] - (input_data[i*dim + j] >= 0)) -
                        	log(1 + exp(input_data[i*dim + j] - 2 * input_data[i*dim + j] * (input_data[i*dim + j] >= 0)));

		}else if(sigmoid_output_data[i*dim+j] >= 0.2 && sigmoid_output_data[i*dim+j] < 0.4){
			
			valid_count2++;
			count_neg2++;
			temp_loss_neg2 -= input_data[i*dim + j] * (target[i*dim+j] - (input_data[i*dim + j] >= 0)) -
                                log(1 + exp(input_data[i*dim + j] - 2 * input_data[i*dim + j] * (input_data[i*dim + j] >= 0)));

		}else if(sigmoid_output_data[i*dim+j] >= 0.4 && sigmoid_output_data[i*dim+j] < 0.6){
			
			valid_count3++;
			count_neg3++;
			temp_loss_neg3 -= input_data[i*dim + j] * (target[i*dim+j] - (input_data[i*dim + j] >= 0)) -
                                log(1 + exp(input_data[i*dim + j] - 2 * input_data[i*dim + j] * (input_data[i*dim + j] >= 0)));

		}else if(sigmoid_output_data[i*dim+j] >= 0.6 && sigmoid_output_data[i*dim+j] < 0.8){
			
			valid_count4++;
			count_neg4++;
			temp_loss_neg4 -= input_data[i*dim + j] * (target[i*dim+j] - (input_data[i*dim + j] >= 0)) -
                                log(1 + exp(input_data[i*dim + j] - 2 * input_data[i*dim + j] * (input_data[i*dim + j] >= 0)));

		}else if(sigmoid_output_data[i*dim+j] >= 0.8 && sigmoid_output_data[i*dim+j] <= 1.0){
			
			valid_count5++;
			count_neg5++;
			temp_loss_neg5 -= input_data[i*dim + j] * (target[i*dim+j] - (input_data[i*dim + j] >= 0)) -
                                log(1 + exp(input_data[i*dim + j] - 2 * input_data[i*dim + j] * (input_data[i*dim + j] >= 0)));

		}
		//
		//count_neg ++;
        	//temp_loss_neg -= input_data[i*dim + j] * (target[i*dim+j] - (input_data[i*dim + j] >= 0)) -
                	//log(1 + exp(input_data[i*dim + j] - 2 * input_data[i*dim + j] * (input_data[i*dim + j] >= 0)));
    	}
	//++valid_count;
     } 
     //liuteng - tongji bili
     temp_count_neg = count_pos * key_;
     if(temp_count_neg == 0){
        //0  de hua ying jia quan bu  fang zhi count_pos + count_neg =0
        valid_count = valid_count5+valid_count4+valid_count3+valid_count2+valid_count1+valid_count0;
        count_neg =count_neg5+count_neg4+count_neg3+count_neg2+count_neg1;
        temp_loss_neg = temp_loss_neg5+temp_loss_neg4+temp_loss_neg3+temp_loss_neg2+temp_loss_neg1;
        loss_neg += temp_loss_neg * count_pos / (count_pos + count_neg);

     }else if(temp_count_neg <= count_neg5){
	
	valid_count = valid_count5+valid_count0;
	count_neg = count_neg5;
        temp_loss_neg = temp_loss_neg5;
	loss_neg += temp_loss_neg * count_pos / (count_pos + count_neg);
 
    }else if(temp_count_neg <= (count_neg5 + count_neg4)){
	
	valid_count = valid_count5+valid_count4+valid_count0;
	count_neg = count_neg5+count_neg4;
	temp_loss_neg = temp_loss_neg5+temp_loss_neg4;
	loss_neg += temp_loss_neg * count_pos / (count_pos + count_neg);	

    }else if(temp_count_neg <= (count_neg5 + count_neg4 + count_neg3)){
	
	valid_count = valid_count5+valid_count4+valid_count3+valid_count0;
	count_neg = count_neg5+count_neg4+count_neg3;
	temp_loss_neg = temp_loss_neg5+temp_loss_neg4+temp_loss_neg3;
	loss_neg += temp_loss_neg * count_pos / (count_pos + count_neg);

    }else if(temp_count_neg <= (count_neg5 + count_neg4 + count_neg3 + count_neg2)){
	
	valid_count = valid_count5+valid_count4+valid_count3+valid_count2+valid_count0;
	count_neg = count_neg5+count_neg4+count_neg3+count_neg2;
	temp_loss_neg = temp_loss_neg5+temp_loss_neg4+temp_loss_neg3+temp_loss_neg2;
	loss_neg += temp_loss_neg * count_pos / (count_pos + count_neg);
   
    }else{
	
	valid_count = valid_count5+valid_count4+valid_count3+valid_count2+valid_count1+valid_count0;
   	count_neg =count_neg5+count_neg4+count_neg3+count_neg2+count_neg1;
	temp_loss_neg = temp_loss_neg5+temp_loss_neg4+temp_loss_neg3+temp_loss_neg2+temp_loss_neg1;
	loss_neg += temp_loss_neg * count_pos / (count_pos + count_neg);
   }
     //
     loss_pos += temp_loss_pos * lambda_ * count_neg / (count_pos + count_neg);
    // loss_neg += temp_loss_neg * count_pos / (count_pos + count_neg);
  }

#if 0
  summ = 0;
  summ = count_pos + count_neg;
  LOG(INFO) <<"_______________________________________________________________";
  LOG(INFO) << "valid_count: " << valid_count <<",valid_count0:"<< valid_count0<<",valid_count1:"<<valid_count1<<",valid_count2:"<<valid_count2<<",valid_count3:"<<valid_count3<<",valid_count4:"<<valid_count4<<",valid_count5:"<<valid_count5;
  LOG(INFO) <<"count_neg1:"<<count_neg1<<",count_neg2:"<<count_neg2<<",count_neg3:"<<count_neg3<<",count_neg4:"<<count_neg4<<",count_neg5:"<<count_neg5;
  LOG(INFO) << "count_pos: " << count_pos << "count_neg: " << "," << count_neg;
  LOG(INFO) << "summ: " << summ;
  LOG(INFO) << "num: " << num;
  LOG(INFO) << "dim: " << dim;
  LOG(INFO) << "temp_loss_neg:"<<temp_loss_neg<<",temp_loss_pos:"<<temp_loss_pos;
  LOG(INFO) << "temp_loss_neg1:"<<temp_loss_neg1<<",temp_loss_neg2:"<<temp_loss_neg2<<",temp_loss_neg3:"<<temp_loss_neg3<<",temp_loss_neg4:"<<temp_loss_neg4<<",temp_loss_neg5"<<temp_loss_neg5;
  LOG(INFO) << "loss_pos: " << loss_pos;
  LOG(INFO) << "loss_neg: " << loss_neg;
  LOG(INFO) <<"______________________________________________________________";
#endif

  loss = (loss_pos * 1 + loss_neg) / num;
  top[0]->mutable_cpu_data()[0] = loss; 
}

template <typename Dtype>
void BinLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    Dtype count_pos = 0;
    Dtype count_neg = 0;
    int dim = bottom[0]->count() / bottom[0]->num();
    const int num = bottom[0]->num();

    Dtype count_neg1 = 0;
    Dtype count_neg2 = 0;
    Dtype count_neg3 = 0;
    Dtype count_neg4 = 0;
    Dtype count_neg5 = 0;
    Dtype temp_count_neg = 0;
    //
    for (int i = 0; i < num; ++i) {
    	count_pos = 0;
    	count_neg = 0;
        //liuteng -  tong ji 5 ge tong de zhi
        count_neg1 =0;
	count_neg2 =0;
	count_neg3 =0;
 	count_neg4 =0;
 	count_neg5 =0;
	temp_count_neg = 0;
	//
        
    	for (int j = 0; j < dim; j ++) {
           	if (target[i*dim+j] == 1) {
                	count_pos ++;
        	}
        	else if (target[i*dim+j] == 0) {
                      //	count_neg++;
		     //liuteng --
                      if(sigmoid_output_data[i*dim+j] >= 0 && sigmoid_output_data[i*dim+j] < 0.2){

                      		  count_neg1++;

               		 }else if(sigmoid_output_data[i*dim+j] >= 0.2 && sigmoid_output_data[i*dim+j] < 0.4){

                        	  count_neg2++;

                	 }else if(sigmoid_output_data[i*dim+j] >= 0.4 && sigmoid_output_data[i*dim+j] < 0.6){

                       		  count_neg3++;

              		 }else if(sigmoid_output_data[i*dim+j] >= 0.6 && sigmoid_output_data[i*dim+j] < 0.8){

	                         count_neg4++;

               		 }else if(sigmoid_output_data[i*dim+j] >= 0.8 && sigmoid_output_data[i*dim+j] <= 1.0){

                       	          count_neg5++;

                	}

 		     //
        	}
     	}
	//liuteng 
	temp_count_neg = count_pos * key_;

	if(temp_count_neg == 0){
		
		count_neg = count_neg5+count_neg4+count_neg3+count_neg2+count_neg1;	
	
	
	}else if(temp_count_neg <= count_neg5){

		count_neg = count_neg5;

	}else if(temp_count_neg <= (count_neg5+count_neg4)){

		count_neg = count_neg5+count_neg4;	

	}else if(temp_count_neg <= (count_neg5+count_neg4+count_neg3)){

		count_neg = count_neg5+count_neg4+count_neg3;	

	}else if(temp_count_neg <= (count_neg5+count_neg4+count_neg3+count_neg2)){

		count_neg = count_neg5+count_neg4+count_neg3+count_neg2;

	}else{

		count_neg = count_neg5+count_neg4+count_neg3+count_neg2+count_neg1;
	
	}
	//
    	for (int j = 0; j < dim; j ++) {
		if (has_ignore_label_) {
			const int target_value = static_cast<int>(target[i*dim+j]);
			if (target_value == ignore_label_) {
         		 bottom_diff[i * dim + j] = 0;
			}
		}
        	if (target[i*dim+j] == 1) {

               		bottom_diff[i * dim + j] *= lambda_ * count_neg / (count_pos + count_neg);
			
        	}
        	else if (target[i*dim+j] == 0) {
			
			if(temp_count_neg  == 0){
			
				if(sigmoid_output_data[i*dim+j] >=0 && sigmoid_output_data[i*dim+j] <= 1.0){

					bottom_diff[i*dim+j] *= count_pos /(count_pos + count_neg);
				}else{
					bottom_diff[i*dim+j] = 0;
				}		

			}else if(temp_count_neg <= count_neg5){
				if(sigmoid_output_data[i*dim+j] >= 0.8 && sigmoid_output_data[i*dim+j] <= 1.0){
					
					bottom_diff[i * dim + j] *= count_pos / (count_pos + count_neg);
				}else{
					bottom_diff[i * dim + j] =0;
				}
			}else if(temp_count_neg <= (count_neg5+count_neg4)){
				if(sigmoid_output_data[i*dim+j] >= 0.6 && sigmoid_output_data[i*dim+j] <= 1.0){

					bottom_diff[i * dim + j] *= count_pos / (count_pos + count_neg);
				}else{
					bottom_diff[i * dim +j] =0;
				}	
				
			}else if(temp_count_neg <= (count_neg5+count_neg4+count_neg3)){
				if(sigmoid_output_data[i*dim+j] >= 0.4 && sigmoid_output_data[i*dim+j] <= 1.0){
					
					bottom_diff[i * dim + j] *= count_pos / (count_pos + count_neg);
				}else{
					bottom_diff[i * dim +j] =0;
				}

			}else if(temp_count_neg <= (count_neg5+count_neg4+count_neg3+count_neg2)){
				if(sigmoid_output_data[i*dim+j] >= 0.2 && sigmoid_output_data[i*dim+j] <= 1.0){
					
					bottom_diff[i * dim + j] *= count_pos / (count_pos + count_neg);
				}else{
					bottom_diff[i * dim + j] = 0;
				}
			
			}else{
                		bottom_diff[i * dim + j] *= count_pos / (count_pos + count_neg);
			}
        	}
     	}
    }
    const Dtype loss_weight = top [0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(BinLossLayer);
#endif

INSTANTIATE_CLASS(BinLossLayer);
REGISTER_LAYER_CLASS(BinLoss);

}  // namespace caffe
