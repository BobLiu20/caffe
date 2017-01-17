// This program Compute the mean_image of a set of images given by a file list
// Usage:
//   compute_imageset_mean [FLAGS] ROOTFOLDER/ LISTFILE MEAN_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <stdint.h>
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Compute the mean_image of a set of images given by"
        " a file list\n"
        "Usage:\n"
        "    compute_imageset_mean [FLAGS] ROOTFOLDER/ LISTFILE MEAN_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_imageset_mean");
    return 1;
  }

  const bool is_color = !FLAGS_gray;

  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::string, int> > lines;
  std::string line;
  size_t pos;
  int label;
  while (std::getline(infile, line)) {
    pos = line.find_last_of(' ');
    label = atoi(line.substr(pos + 1).c_str());
    lines.push_back(std::make_pair(line.substr(0, pos), label));
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  BlobProto sum_blob;
  std::string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  int size_in_datum = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;
    status = ReadImageToDatum(root_folder + lines[line_id].first,
        lines[line_id].second, resize_height, resize_width, is_color,
        "", &datum);
    if (status == false) continue;
    if (!data_size_initialized) {
      data_size_initialized = true;
      sum_blob.set_num(1);
      sum_blob.set_channels(datum.channels());
      sum_blob.set_height(datum.height());
      sum_blob.set_width(datum.width());
      size_in_datum = std::max<int>(datum.data().size(),
                                        datum.float_data_size());
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.add_data(0.);
      }
    } else {
      const std::string& data = datum.data();
      if (data.size() != 0) {
        CHECK_EQ(data.size(), size_in_datum);
        for (int i = 0; i < size_in_datum; ++i) {
          sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
        }
      } else {
        CHECK_EQ(datum.float_data_size(), size_in_datum);
        for (int i = 0; i < size_in_datum; ++i) {
          sum_blob.set_data(i, sum_blob.data(i) +
              static_cast<float>(datum.float_data(i)));
        }
      }
      ++count;
      if (count % 10000 == 0) {
        LOG(INFO) << "Processed " << count << " files.";
      }
    }
  }
  if (count % 10000 != 0) {
    LOG(INFO) << "Processed " << count << " files.";
  }
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / count);
  }
  // Write to disk
  if (argc == 4) {
    LOG(INFO) << "Write to " << argv[3];
    WriteProtoToBinaryFile(sum_blob, argv[3]);
  }
  const int channels = sum_blob.channels();
  const int dim = sum_blob.height() * sum_blob.width();
  std::vector<float> mean_values(channels, 0.0);
  LOG(INFO) << "Number of channels: " << channels;
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < dim; ++i) {
      mean_values[c] += sum_blob.data(dim * c + i);
    }
    LOG(INFO) << "mean_value channel [" << c << "]:" << mean_values[c] / dim;
  }

#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
