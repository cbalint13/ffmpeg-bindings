#ifndef FFMPEG_VIDEO_H
#define FFMPEG_VIDEO_H

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// FFmpeg headers
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/mem.h>
#include <libavutil/opt.h>
}

// OpenCV headers
#include <opencv2/opencv.hpp>

class FFMPEGVideo {
public:
  FFMPEGVideo(const std::string &filename, const std::string &filter_descr_str);
  ~FFMPEGVideo();

  bool isInitialized() const;
  bool GetNextFrame(cv::Mat &output_mat);

  // Getter methods
  int get_frame_width() const;
  int get_frame_height() const;
  int get_frame_id() const;
  int get_frame_total() const;
  int64_t get_last_frame_pts() const;
  double get_last_frame_time_seconds() const;

private:
  std::string input_filename_;
  std::string filter_descr_;

  AVFormatContext *fmt_ctx;
  AVCodecContext *dec_ctx;
  AVFilterGraph *filter_graph;
  AVFilterContext *buffersrc_ctx;
  AVFilterContext *buffersink_ctx;
  AVBufferRef *hw_device_ctx;
  AVBufferRef *hw_frames_ctx;
  AVPacket *pkt;
  AVFrame *frame;
  AVFrame *filt_frame;
  int video_stream_idx;
  bool initialized;

  int frame_count_;
  int total_frames_;
  int frame_width_;
  int frame_height_;
  AVRational video_time_base_;

  int64_t current_frame_pts_;
  double current_frame_time_seconds_;

  // Private helper function to handle a successfully retrieved filtered frame.
  bool process_retrieved_frame(cv::Mat &output_mat_ref);

  // Callback for hardware format negotiation (static member function)
  static enum AVPixelFormat get_hw_format(AVCodecContext *ctx,
                                          const enum AVPixelFormat *pix_fmts);
  // Initialization and cleanup methods
  bool init();
  void cleanup();
};

#endif // FFMPEG_VIDEO_H
