#include <errno.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ffmpeg_video.h"

namespace py = pybind11;

// Function to convert cv::Mat to py::array_t (NumPy array) by copying data
py::array_t<uint8_t> mat_to_numpy(const cv::Mat &mat) {
  if (!mat.isContinuous() || mat.depth() != CV_8U ||
      (mat.channels() != 1 && mat.channels() != 3)) {
    throw std::runtime_error("Unsupported Mat format for numpy conversion.");
  }

  // Determine shape and strides based on the number of channels
  std::vector<ssize_t> shape;
  std::vector<ssize_t> strides;

  if (mat.channels() == 1) { // Grayscale
    shape = {mat.rows, mat.cols};
    strides = {static_cast<ssize_t>(mat.step[0]),  // row stride
               static_cast<ssize_t>(mat.step[1])}; // col stride (pixel stride)
  } else {                                         // Color (e.g., BGR24)
    shape = {mat.rows, mat.cols, mat.channels()};
    strides = {static_cast<ssize_t>(mat.step[0]),      // row stride
               static_cast<ssize_t>(mat.step[1]),      // col stride
               static_cast<ssize_t>(mat.elemSize1())}; // channel stride
  }

  // Create a new py::array_t by copying the data
  // The 'mat.data' pointer is used as the source for the copy.
  // py::array_t's constructor without the 'owner' argument means copy.
  py::array_t<uint8_t> result_array(shape, strides, mat.data);

  return result_array;
}

PYBIND11_MODULE(ffmpeg_video, m) {
  m.doc() = "pybind11 plugin for FFMPEGVideo class";

  py::class_<FFMPEGVideo>(m, "FFMPEGVideo")
      .def(py::init<const std::string &, const std::string &>(),
           py::arg("filename"),
           py::arg("filter_descr_str") = "", // Default empty string
           "Initializes the FFMPEGVideo processor with a video file and an "
           "optional filter graph description.")
      .def("is_initialized", &FFMPEGVideo::isInitialized,
           "Checks if the video processor was successfully initialized.")
      .def("get_frame_width", &FFMPEGVideo::get_frame_width,
           "Returns the width of the output video frames.")
      .def("get_frame_height", &FFMPEGVideo::get_frame_height,
           "Returns the height of the output video frames.")
      .def("get_frame_id", &FFMPEGVideo::get_frame_id,
           "Returns the current frame ID (count of frames successfully "
           "retrieved).")
      .def("get_frame_total", &FFMPEGVideo::get_frame_total,
           "Returns the total number of frames in the video (may be an "
           "approximation).")
      .def("get_last_frame_pts", &FFMPEGVideo::get_last_frame_pts,
           "Returns the Presentation Timestamp (PTS) of the last retrieved "
           "frame.")
      .def("get_last_frame_time_seconds",
           &FFMPEGVideo::get_last_frame_time_seconds,
           "Returns the time in seconds of the last retrieved frame's PTS.")
      .def(
          "get_next_frame",
          [](FFMPEGVideo &self) -> py::object {
            cv::Mat frame_mat;
            if (self.GetNextFrame(frame_mat)) {
              return mat_to_numpy(frame_mat);
            }
            return py::none(); // Return None if no frame is available (EOF or
                               // error)
          },
          "Retrieves the next video frame as a NumPy array (uint8, BGR or "
          "Grayscale). Returns None if the end of the stream is reached or "
          "an error occurs.");
}
