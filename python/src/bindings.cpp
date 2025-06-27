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

// Function to convert cv::Mat to py::array_t (NumPy array)
py::array_t<uint8_t> mat_to_numpy(const cv::Mat &mat) {
  if (!mat.isContinuous()) {
    throw std::runtime_error("Mat is not continuous. Cannot convert to numpy "
                             "array without copying.");
  }

  if (mat.type() != CV_8UC3) {
    throw std::runtime_error("Unsupported Mat type. Expected CV_8UC3.");
  }

  // Shape of the NumPy array (height, width, channels)
  std::vector<ssize_t> shape = {mat.rows, mat.cols, mat.channels()};
  // Strides (how many bytes to move to get to the next element in each
  // dimension)
  std::vector<ssize_t> strides = {
      static_cast<ssize_t>(mat.step[0]),    // row stride
      static_cast<ssize_t>(mat.step[1]),    // col stride
      static_cast<ssize_t>(mat.elemSize1()) // channel stride
  };

  return py::array_t<uint8_t>(
      shape, strides, mat.data,
      py::none()); // py::none() for base object means no external ownership
}

PYBIND11_MODULE(ffmpeg_video, m) {
  m.doc() =
      "pybind11 plugin for FFMPEGVideo class with RKMpp hardware acceleration";

  py::class_<FFMPEGVideo>(m, "FFMPEGVideo")
      // Updated constructor binding with a default filter_descr
      .def(py::init<const std::string &, const std::string &>(),
           py::arg("filename"),
           py::arg("filter_descr") = "scale_rkrga=w=1280:h=720:format=bgr24,"
                                     "hwmap=mode=read,format=bgr24",
           "Initializes the FFMPEGVideo processor with the given video "
           "filename and an optional filter description string.")
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
          "Fetches the next processed video frame as a NumPy array. Returns "
          "None if EOF or error.");
}
