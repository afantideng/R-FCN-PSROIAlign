// ------------------------------------------------------------------
// R-FCN
// Written by Jiangfan Deng
// ------------------------------------------------------------------

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/psroi_align_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {
  template <typename Dtype>
  void PSROIAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    PSROIAlignParameter psroi_align_param =
      this->layer_param_.psroi_align_param();
    spatial_scale_ = psroi_align_param.spatial_scale();
    LOG(INFO) << "Spatial scale: " << spatial_scale_;

    CHECK_GT(psroi_align_param.output_dim(), 0)
      << "output_dim must be > 0";
    CHECK_GT(psroi_align_param.group_size(), 0)
      << "group_size must be > 0";

    output_dim_ = psroi_align_param.output_dim();
    group_size_ = psroi_align_param.group_size();
    sample_num_ = psroi_align_param.sample_num();
    pooled_height_ = group_size_;
    pooled_width_ = group_size_;
  }

  template <typename Dtype>
  void PSROIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    channels_ = bottom[0]->channels();
    CHECK_EQ(channels_, output_dim_*group_size_*group_size_)
      << "input channel number does not match layer parameters";
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    top[0]->Reshape(bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
    mapping_channel_.Reshape(bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
    sample_pos_.Reshape(bottom[1]->num(), output_dim_, pooled_height_*pooled_width_*sample_num_*sample_num_, 2);
  }

  template <typename Dtype>
  void PSROIAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
  }

  template <typename Dtype>
  void PSROIAlignLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
#ifdef CPU_ONLY
  STUB_GPU(PSROIAlignLayer);
#endif

  INSTANTIATE_CLASS(PSROIAlignLayer);
  REGISTER_LAYER_CLASS(PSROIAlign);

}  // namespace caffe
