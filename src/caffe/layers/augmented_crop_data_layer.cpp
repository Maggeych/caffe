#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_int.hpp"
#include "boost/random/uniform_real.hpp"
#include <boost/random/random_device.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/augmented_crop_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

class AugmentedCropWarp {
 public:
  AugmentedCropWarp(const AugmentedCropDataParameter& p) : rng_seed(dev) {
    // Get unaugmented scaling and translation so the ROI defined in the
    // parameters has target size and is centered in the origin.
    cv::Point2f tl(p.crop_x1(), p.crop_y1());  // Top left.
    cv::Point2f br(p.crop_x2(), p.crop_y2());  // Bottom right.

    cv::Point2f roiDim = br - tl;
    this->targetDim = cv::Size(p.out_width(), p.out_height());
    if (this->targetDim.width == 0.0f) this->targetDim.width = roiDim.x;
    if (this->targetDim.height == 0.0f) this->targetDim.height = roiDim.y;
    this->scale = cv::Point2f(
        // roiDim.x / this->targetDim.width,
        // roiDim.y / this->targetDim.height
        this->targetDim.width / roiDim.x,
        this->targetDim.height / roiDim.y
        );
    this->translate = (tl + br) * 0.5f;

    // Create the distributions with variances according to the parameters.
    if ((this->use_fx = (p.var_fx() > 0.0)))
      this->rng_fx = boost::random::uniform_real_distribution<>(
          1.0 - p.var_fx(), 1.0 + p.var_fx()
          );
    if ((this->use_fy = (p.var_fy() > 0.0)))
      this->rng_fy = boost::random::uniform_real_distribution<>(
          1.0 - p.var_fy(), 1.0 + p.var_fy()
          );
    if ((this->use_sx = (p.var_sx() > 0.0)))
      this->rng_sx = boost::random::uniform_real_distribution<>(
          0.0 - p.var_sx(), 0.0 + p.var_sx()
          );
    if ((this->use_sy = (p.var_sy() > 0.0)))
      this->rng_sy = boost::random::uniform_real_distribution<>(
          0.0 - p.var_sy(), 0.0 + p.var_sy()
          );
    if ((this->use_tx = (p.var_tx() > 0.0)))
      this->rng_tx = boost::random::uniform_real_distribution<>(
          -p.var_tx(), p.var_tx()
          );
    if ((this->use_ty = (p.var_ty() > 0.0)))
      this->rng_ty = boost::random::uniform_real_distribution<>(
          -p.var_ty(), p.var_ty()
          );
    if ((this->use_brightness = (p.var_brightness() > 0.0)))
      this->rng_brightness = boost::random::uniform_int_distribution<>(
          -p.var_brightness(), p.var_brightness()
          );
    if ((this->use_rotation = (p.var_rotation() > 0.0)))
      this->rng_rotation = boost::random::uniform_real_distribution<>(
          -p.var_rotation(), p.var_rotation()
          );
  }

  cv::Mat warpAugmented(const cv::Mat& in) {
    cv::Mat out;
    cv::warpAffine(in, out, this->sample(), this->targetDim);

    //cv::imwrite("warped.bmp", out);
    // cv::imshow("warped", out);
    // cv::waitKey(1);
    return out;
  }

  cv::Mat sample() {
    // Get augmented scaling to the target size.
    cv::Point2f f = this->f();
    // Translation variance is defined in the unscaled pixel distance space.
    // In the homogeneous matrix transformation it is applied after the
    // scaling, so the augmented translation needs to also be scaled
    // accordingly.
    cv::Point2f t = this->t();
    t.x *= f.x; t.y *= f.y;
    // The final augmented homogeneous scale and translation matrix.
    cv::Mat P = cv::Mat::eye(3, 3, CV_32F);
    P.at<float>(0, 0) = f.x; P.at<float>(1, 1) = f.y;
    P.at<float>(0, 2) = t.x; P.at<float>(1, 2) = t.y;

    // Shear matrix.
    cv::Mat S = cv::Mat::eye(3, 3, CV_32F);
    S.at<float>(0, 1) = this->sx();
    S.at<float>(1, 0) = this->sy();

    // 2x3 Rotation matrix.
    cv::Mat rot = cv::getRotationMatrix2D(
        cv::Point2f(0.0, 0.0),  // Center of rotation.
        this->rotation(),  // Angle.
        1.0  // Scale.
        );
    // Make it 3x3.
    cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
    rot.copyTo(R(cv::Rect(0, 0, 3, 2)));

    // ROI is still centered in the origin. Move so the ROIs top left corner is
    // the origin.
    cv::Mat B = cv::Mat::eye(3, 3, CV_32F);
    B.at<float>(0, 2) = this->targetDim.width * 0.5;
    B.at<float>(1, 2) = this->targetDim.height * 0.5;

    // First scale and move ROI center to origin (P), shear (S),
    // then rotate (R) and finally move the top left ROI corner to (0,0).
    return (B * R * S * P)(cv::Rect(0, 0, 3, 2));
  }

 protected:
  inline cv::Point2f f() { return cv::Point2f(fx(), fy()); }
  inline float fx() {
    if (use_fx) return this->rng_fx(this->rng_seed) * this->scale.x;
    else return this->scale.x;
  }
  inline float fy() {
    if (use_fy) return this->rng_fy(this->rng_seed) * this->scale.y;
    else return this->scale.y;
  }

  inline cv::Point2f s() { return cv::Point2f(sx(), sy()); }
  inline float sx() {
    if (use_sx) return this->rng_sx(this->rng_seed);
    else return 0.0f;
  }
  inline float sy() {
    if (use_sy) return this->rng_sy(this->rng_seed);
    else return 0.0f;
  }

  inline cv::Point2f t() { return cv::Point2f(tx(), ty()); }
  inline float tx() {
    if (use_tx) return this->rng_tx(this->rng_seed) - this->translate.x;
    else return - this->translate.x;
  }
  inline float ty() {
    if (use_ty) return this->rng_ty(this->rng_seed) - this->translate.y;
    else return - this->translate.y;
  }

  inline float rotation() {
    if (use_rotation) return this->rng_rotation(this->rng_seed);
    else return 0.0f;
  }

  inline float brightness() {
    if (use_brightness) return this->rng_brightness(this->rng_seed);
    else return 0.0f;
  }

  cv::Point2f scale;
  cv::Point2f translate;
  cv::Size targetDim;

  bool use_fx, use_fy, use_sx, use_sy, use_tx, use_ty, use_rotation,
       use_brightness;
  boost::random_device dev;
  boost::random::mt19937 rng_seed;
  boost::random::uniform_real_distribution<> rng_fx;
  boost::random::uniform_real_distribution<> rng_fy;
  boost::random::uniform_real_distribution<> rng_sx;
  boost::random::uniform_real_distribution<> rng_sy;
  boost::random::uniform_real_distribution<> rng_tx;
  boost::random::uniform_real_distribution<> rng_ty;
  boost::random::uniform_real_distribution<> rng_rotation;
  boost::random::uniform_int_distribution<> rng_brightness;
};

template <typename Dtype>
AugmentedCropDataLayer<Dtype>::~AugmentedCropDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void AugmentedCropDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Get the ImageCrop parameter and retrieve all settings from it.
  const AugmentedCropDataParameter& iidp =
    this->layer_param_.augmented_crop_param();
  CHECK(
      (iidp.out_width() == 0 && iidp.out_height() == 0) ||
      (iidp.out_width() > 0 && iidp.out_height() > 0)
      )
    << "Current implementation requires out_height and out_width to be set at "
    "the same time.";

  const string& source = iidp.images();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  while (std::getline(infile, line)) {
    image_paths.push_back(line);
      cv::Mat img = ReadImageToCVMat(iidp.root_folder() + line, 0, 0, iidp.color());
      CHECK(img.data) << "Could not load " << line;
      images.push_back(img);
  }

  CHECK(!image_paths.empty()) << "File is empty";

  if (iidp.shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  } else {
    if (this->phase_ == TRAIN && Caffe::solver_rank() > 0) {
      LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
    }
  }
  LOG(INFO) << "A total of " << image_paths.size() << " images.";

  // Read an image, and use it to initialize the top blob.
  current_image = 0;
  cv::Mat cv_img = ReadImageToCVMat(
      iidp.root_folder() + image_paths[current_image],
      iidp.out_height(), iidp.out_width(), iidp.color());
  CHECK(cv_img.data) << "Could not load " << image_paths[current_image];
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = iidp.batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
}

template <typename Dtype>
void AugmentedCropDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(images.begin(), images.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void AugmentedCropDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const AugmentedCropDataParameter& iidp =
    this->layer_param_.augmented_crop_param();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  // cv::Mat cv_img = ReadImageToCVMat(
  //     iidp.root_folder() + image_paths[current_image],
  //     iidp.out_height(), iidp.out_width(), iidp.color()
  //     );
  // CHECK(cv_img.data) << "Could not load " << image_paths[current_image];
  // Use data_transformer to infer the expected blob shape from a cv_img.
  // vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  vector<int> top_shape(4);
  top_shape[0] = 1;
  top_shape[1] = iidp.color() ? 3 : 1;
  top_shape[2] = iidp.out_height();
  top_shape[3] = iidp.out_width();
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = iidp.batch_size();
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();

  AugmentedCropWarp warp(iidp);
  const int image_cnt = images.size();
  for (int item_id = 0; item_id < iidp.batch_size(); ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(image_cnt, current_image);
    cv::Mat cv_img = images[current_image];
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    // Do augmented cropping.
    cv::Mat augmented = warp.warpAugmented(cv_img);
    this->data_transformer_->Transform(augmented, &(this->transformed_data_));

    trans_time += timer.MicroSeconds();

    // go to the next iter
    current_image++;
    if (current_image >= image_cnt) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      current_image = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(AugmentedCropDataLayer);
REGISTER_LAYER_CLASS(AugmentedCropData);

}  // namespace caffe
#endif  // USE_OPENCV
