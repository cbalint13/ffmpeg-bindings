#include "ffmpeg_video.h"

// Helper function definitions
static std::string av_error_to_string(int errnum) {
  char err_buf[AV_ERROR_MAX_STRING_SIZE];
  av_strerror(errnum, err_buf, sizeof(err_buf));
  return std::string(err_buf);
}

static bool check_error(int ret, const std::string &msg) {
  if (ret < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(ret, errbuf, sizeof(errbuf));
    std::cerr << msg << ": " << errbuf << std::endl;
    return true; // Indicate error
  }
  return false; // Indicate success
}

// FFMPEGVideo Class Implementation
FFMPEGVideo::FFMPEGVideo(const std::string &filename,
                         const std::string &filter_descr_str)
    : input_filename_(filename), filter_descr_(filter_descr_str),
      fmt_ctx(nullptr), dec_ctx(nullptr), filter_graph(nullptr),
      buffersrc_ctx(nullptr), buffersink_ctx(nullptr), hw_device_ctx(nullptr),
      hw_frames_ctx(nullptr), pkt(nullptr), frame(nullptr), filt_frame(nullptr),
      video_stream_idx(-1), initialized(false), frame_count_(0),
      total_frames_(0), frame_width_(0), frame_height_(0),
      video_time_base_({0, 1}), current_frame_pts_(AV_NOPTS_VALUE),
      current_frame_time_seconds_(0.0) {
  pkt = av_packet_alloc();
  frame = av_frame_alloc();
  filt_frame = av_frame_alloc();

  if (!pkt || !frame || !filt_frame) {
    std::cerr << "Failed to allocate AVPacket or AVFrame. Out of memory?"
              << std::endl;
    return;
  }

  initialized = init();
}

FFMPEGVideo::~FFMPEGVideo() { cleanup(); }

bool FFMPEGVideo::isInitialized() const { return initialized; }

// Private helper function to handle a successfully retrieved filtered frame.
bool FFMPEGVideo::process_retrieved_frame(cv::Mat &output_mat_ref) {

  // Determine the OpenCV matrix type based on the pixel format
  int cv_type;
  // Explicitly cast filt_frame->format to AVPixelFormat to resolve the error
  const AVPixFmtDescriptor *desc =
      av_pix_fmt_desc_get(static_cast<AVPixelFormat>(filt_frame->format));
  if (!desc) {
    std::cerr << "Unknown pixel format: " << filt_frame->format << std::endl;
    return false;
  }

  std::cout << "Detected output pixel format: " << desc->name << std::endl;

  if (desc->nb_components == 1) { // Grayscale
    cv_type = CV_8UC1;
  } else if (desc->nb_components == 3) { // Color (e.g., BGR24)
    cv_type = CV_8UC3;
  } else {
    std::cerr << "Unsupported number of components for OpenCV conversion: "
              << desc->nb_components << std::endl;
    return false;
  }

  output_mat_ref = cv::Mat(filt_frame->height, filt_frame->width, cv_type,
                           filt_frame->data[0], filt_frame->linesize[0])
                       .clone();

  if (frame_count_ == 0) { // Only set once for first frame
    frame_width_ = filt_frame->width;
    frame_height_ = filt_frame->height;
  }
  frame_count_++;

  current_frame_pts_ = filt_frame->pts;
  if (video_time_base_.num != 0 && video_time_base_.den != 0) {
    current_frame_time_seconds_ = current_frame_pts_ * av_q2d(video_time_base_);
  } else {
    current_frame_time_seconds_ = 0.0;
  }

  av_frame_unref(filt_frame);
  return true;
}

bool FFMPEGVideo::GetNextFrame(cv::Mat &output_mat) {
  if (!initialized) {
    std::cerr << "FFMPEGVideo not initialized. Cannot get frame." << std::endl;
    return false;
  }

  int ret = 0;
  bool frame_retrieved = false;
  bool end_of_input_reached = false;

  while (!frame_retrieved) {
    ret = av_buffersink_get_frame(buffersink_ctx, filt_frame);
    if (ret >= 0) {
      return process_retrieved_frame(output_mat);
    } else if (ret == AVERROR(EAGAIN)) {
      // Filter graph needs more input. Proceed to decoding/reading.
    } else if (ret == AVERROR_EOF) {
      end_of_input_reached = true;
      break;
    } else {
      check_error(ret, "Error receiving frame from filter graph");
      return false;
    }

    if (!end_of_input_reached) {
      ret = avcodec_receive_frame(dec_ctx, frame);
      if (ret >= 0) {
        frame->pts = frame->best_effort_timestamp;

        int add_frame_ret = av_buffersrc_add_frame_flags(
            buffersrc_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF);
        if (check_error(add_frame_ret, "Error feeding frame to filter graph")) {
          av_frame_unref(frame);
          return false;
        }
        av_frame_unref(frame);
        continue;
      } else if (ret == AVERROR(EAGAIN)) {
        // Decoder needs more packets. Proceed to reading input.
      } else if (ret == AVERROR_EOF) {
        end_of_input_reached = true;
        break;
      } else {
        check_error(ret, "Error receiving frame from decoder");
        return false;
      }

      ret = av_read_frame(fmt_ctx, pkt);
      if (ret < 0) {
        if (ret == AVERROR_EOF) {
          end_of_input_reached = true;
          break;
        } else {
          check_error(ret, "Error reading packet from input");
          return false;
        }
      }

      if (pkt->stream_index == video_stream_idx) {
        int send_packet_ret;
        while ((send_packet_ret = avcodec_send_packet(dec_ctx, pkt)) ==
               AVERROR(EAGAIN)) {
          int drain_ret = avcodec_receive_frame(dec_ctx, frame);
          if (drain_ret >= 0) {
            int add_drain_frame_to_filter_ret = av_buffersrc_add_frame_flags(
                buffersrc_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF);
            if (check_error(add_drain_frame_to_filter_ret,
                            "Error feeding drained frame to filter graph")) {
              av_frame_unref(frame);
              av_packet_unref(pkt);
              return false;
            }
            av_frame_unref(frame);

            int pull_filtered_ret =
                av_buffersink_get_frame(buffersink_ctx, filt_frame);
            if (pull_filtered_ret >= 0) {
              av_packet_unref(pkt);
              return process_retrieved_frame(output_mat);
            } else if (pull_filtered_ret != AVERROR(EAGAIN) &&
                       pull_filtered_ret != AVERROR_EOF) {
              check_error(pull_filtered_ret, "Error receiving filtered frame "
                                             "during packet send retry drain");
              av_packet_unref(pkt);
              return false;
            }
          } else if (drain_ret == AVERROR(EAGAIN) || drain_ret == AVERROR_EOF) {
            std::cerr << "Warning: Cannot drain decoder to send packet. "
                         "Breaking send retry loop."
                      << std::endl;
            break;
          } else {
            check_error(
                drain_ret,
                "Error receiving frame from decoder during packet send retry");
            av_packet_unref(pkt);
            return false;
          }
        }

        if (send_packet_ret < 0) {
          check_error(send_packet_ret,
                      "Error sending packet to decoder after retries");
          av_packet_unref(pkt);
          return false;
        }
      }
      av_packet_unref(pkt);
    }
  } // End of main while(!frame_retrieved) loop

  // --- Final Flushing Logic ---
  if (end_of_input_reached) {
    std::cout << "Initiating pipeline flushing..." << std::endl;

    avcodec_send_packet(dec_ctx, nullptr);
    while (true) {
      ret = avcodec_receive_frame(dec_ctx, frame);
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        break;
      } else if (check_error(ret,
                             "Error receiving flushed frame from decoder")) {
        return false;
      }
      frame->pts = frame->best_effort_timestamp;
      int add_flush_frame_to_filter_ret = av_buffersrc_add_frame_flags(
          buffersrc_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF);
      if (check_error(add_flush_frame_to_filter_ret,
                      "Error feeding flushed frame to filter graph")) {
        av_frame_unref(frame);
        return false;
      }
      av_frame_unref(frame);
    }

    int add_flush_to_buffersrc_ret =
        av_buffersrc_add_frame_flags(buffersrc_ctx, nullptr, 0);
    if (check_error(add_flush_to_buffersrc_ret,
                    "Error flushing buffer source")) {
      return false;
    }

    while (true) {
      ret = av_buffersink_get_frame(buffersink_ctx, filt_frame);
      if (ret >= 0) {
        return process_retrieved_frame(output_mat);
      } else if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        break;
      } else {
        check_error(ret, "Error receiving flushed frame from filter graph");
        return false;
      }
    }
  }
  return false;
}

// Getter implementations
int FFMPEGVideo::get_frame_width() const { return frame_width_; }
int FFMPEGVideo::get_frame_height() const { return frame_height_; }
int FFMPEGVideo::get_frame_id() const { return frame_count_; }
int FFMPEGVideo::get_frame_total() const { return total_frames_; }
int64_t FFMPEGVideo::get_last_frame_pts() const { return current_frame_pts_; }
double FFMPEGVideo::get_last_frame_time_seconds() const {
  return current_frame_time_seconds_;
}

// Callback function to select the hardware pixel format for the decoder
// Note: 'static' is part of the declaration in the header for static member
// functions, but not repeated in the definition outside the class scope.
enum AVPixelFormat
FFMPEGVideo::get_hw_format(AVCodecContext *ctx,
                           const enum AVPixelFormat *pix_fmts) {
  const enum AVPixelFormat *p;
  std::cout << "get_hw_format called. Supported formats by decoder/hw:"
            << std::endl;
  for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
    std::cout << "- " << av_get_pix_fmt_name(*p) << std::endl;
  }

  for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
    if (*p == AV_PIX_FMT_DRM_PRIME) {
      std::cout << "Negotiating HW Pixel Format: AV_PIX_FMT_DRM_PRIME for "
                   "decoder output."
                << std::endl;
      return *p;
    }
  }
  for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
    if (*p == AV_PIX_FMT_NV12) {
      std::cout
          << "Negotiating HW Pixel Format: AV_PIX_FMT_NV12 for decoder output."
          << std::endl;
      return *p;
    }
  }

  std::cerr << "Failed to get required HW surface format (DRM_PRIME or NV12 "
               "not supported by decoder/HW)."
            << std::endl;
  return AV_PIX_FMT_NONE;
}

// Initializes all FFmpeg components
bool FFMPEGVideo::init() {
  int ret = 0;

  // --- 1. Open input file and find stream info ---
  std::cout << "Opening input file: " << input_filename_ << std::endl;
  ret =
      avformat_open_input(&fmt_ctx, input_filename_.c_str(), nullptr, nullptr);
  if (check_error(ret, "Failed to open input file")) {
    return false;
  }

  std::cout << "Finding stream information..." << std::endl;
  ret = avformat_find_stream_info(fmt_ctx, nullptr);
  if (check_error(ret, "Failed to find stream information")) {
    return false;
  }

  // Find the first video stream
  std::cout << "Finding video stream..." << std::endl;
  video_stream_idx =
      av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (video_stream_idx < 0) {
    std::cerr << "Could not find a video stream in the input file."
              << std::endl;
    return false;
  }

  // Store video stream time_base
  video_time_base_ = fmt_ctx->streams[video_stream_idx]->time_base;

  // Calculate total frames if available or approximate
  if (fmt_ctx->streams[video_stream_idx]->nb_frames > 0) {
    total_frames_ = fmt_ctx->streams[video_stream_idx]->nb_frames;
  } else if (fmt_ctx->streams[video_stream_idx]->duration != AV_NOPTS_VALUE &&
             fmt_ctx->streams[video_stream_idx]->avg_frame_rate.num > 0) {
    total_frames_ = static_cast<int>(
        fmt_ctx->streams[video_stream_idx]->duration *
        av_q2d(fmt_ctx->streams[video_stream_idx]->avg_frame_rate) / 1000);
    if (total_frames_ == 0) {
      std::cerr << "Warning: Total frames could not be accurately determined."
                << std::endl;
    }
  } else {
    std::cerr << "Warning: Total frames count is unavailable or unreliable for "
                 "this stream."
              << std::endl;
  }

  // --- 2. Initialize Hardware Acceleration ---
  AVHWDeviceType hw_type = AV_HWDEVICE_TYPE_NONE;
  const char *hw_device_type_name = "rkmpp";

  hw_type = av_hwdevice_find_type_by_name(hw_device_type_name);
  if (hw_type == AV_HWDEVICE_TYPE_NONE) {
    std::cerr << "Hardware device type '" << hw_device_type_name
              << "' not found. "
              << "This may mean your FFmpeg build does not support it, "
              << "or it's not correctly configured on your system."
              << std::endl;
    return false;
  }
  std::cout << "Using hardware device type: "
            << av_hwdevice_get_type_name(hw_type) << std::endl;

  ret = av_hwdevice_ctx_create(&hw_device_ctx, hw_type, nullptr, nullptr, 0);
  if (check_error(ret, "Failed to create HW device context")) {
    return false;
  }
  std::cout << "Successfully created HW device context: " << hw_device_type_name
            << std::endl;

  // --- 3. Setup Decoder Context ---
  const AVCodec *decoder = avcodec_find_decoder_by_name("hevc_rkmpp");
  if (!decoder) {
    std::cerr << "HEVC rkmpp decoder not found. Ensure FFmpeg is built with "
                 "rkmpp support."
              << std::endl;
    return false;
  }
  std::cout << "Using decoder: " << decoder->name << std::endl;

  dec_ctx = avcodec_alloc_context3(decoder);
  if (!dec_ctx) {
    std::cerr << "Failed to allocate decoder context." << std::endl;
    return false;
  }

  ret = avcodec_parameters_to_context(
      dec_ctx, fmt_ctx->streams[video_stream_idx]->codecpar);
  if (check_error(ret, "Failed to copy codec parameters to decoder context")) {
    return false;
  }

  dec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
  if (!dec_ctx->hw_device_ctx) {
    std::cerr << "Failed to set HW device context for codec context."
              << std::endl;
    return false;
  }
  std::cout << "Hardware device context set for codec context." << std::endl;

  dec_ctx->get_format = get_hw_format;
  std::cout << "Hardware pixel format negotiation callback set." << std::endl;

  std::cout << "Opening decoder..." << std::endl;
  ret = avcodec_open2(dec_ctx, decoder, nullptr);
  if (check_error(ret, "Failed to open decoder")) {
    return false;
  }
  std::cout << "Codec opened successfully. Decoder output pix_fmt: "
            << av_get_pix_fmt_name(dec_ctx->pix_fmt) << std::endl;

  // --- Manually allocate and initialize hw_frames_ctx ---
  hw_frames_ctx = av_hwframe_ctx_alloc(hw_device_ctx);
  if (!hw_frames_ctx) {
    std::cerr << "Failed to allocate AVHWFramesContext." << std::endl;
    return false;
  }

  AVHWFramesContext *frames_ctx_data =
      (AVHWFramesContext *)(hw_frames_ctx->data);
  frames_ctx_data->format = dec_ctx->pix_fmt;
  frames_ctx_data->sw_format = AV_PIX_FMT_NV12;
  frames_ctx_data->width = dec_ctx->width;
  frames_ctx_data->height = dec_ctx->height;
  frames_ctx_data->initial_pool_size = 0;

  ret = av_hwframe_ctx_init(hw_frames_ctx);
  if (check_error(ret, "Failed to initialize AVHWFramesContext")) {
    return false;
  }
  std::cout << "Successfully initialized AVHWFramesContext for decoder's "
               "output (format: "
            << av_get_pix_fmt_name(frames_ctx_data->format) << ")."
            << std::endl;

  av_buffer_unref(&dec_ctx->hw_frames_ctx);
  dec_ctx->hw_frames_ctx = av_buffer_ref(hw_frames_ctx);
  if (!dec_ctx->hw_frames_ctx) {
    std::cerr << "Failed to assign allocated hw_frames_ctx to decoder context "
                 "after init."
              << std::endl;
    return false;
  }
  std::cout << "Assigned explicit hw_frames_ctx to decoder context."
            << std::endl;

  // --- 4. Setup Filter Graph ---
  filter_graph = avfilter_graph_alloc();
  if (!filter_graph) {
    std::cerr << "Failed to allocate filter graph." << std::endl;
    return false;
  }

  AVRational time_base = fmt_ctx->streams[video_stream_idx]->time_base;
  std::string buffersrc_args =
      "video_size=" + std::to_string(dec_ctx->width) + "x" +
      std::to_string(dec_ctx->height) +
      ":pix_fmt=" + av_get_pix_fmt_name(dec_ctx->pix_fmt) +
      ":time_base=" + std::to_string(time_base.num) + "/" +
      std::to_string(time_base.den) +
      ":pixel_aspect=" + std::to_string(dec_ctx->sample_aspect_ratio.num) +
      "/" + std::to_string(dec_ctx->sample_aspect_ratio.den);

  // Use the member variable filter_descr_ here
  // std::string filter_descr =
  // "scale_rkrga=w=1280:h=720:format=bgr24,hwmap=mode=read,format=bgr24"; //
  // Old hardcoded

  const AVFilter *buffersrc = avfilter_get_by_name("buffer");
  const AVFilter *buffersink = avfilter_get_by_name("buffersink");

  ret = avfilter_graph_create_filter(&buffersrc_ctx, buffersrc, "in",
                                     buffersrc_args.c_str(), nullptr,
                                     filter_graph);
  if (check_error(ret, "Cannot create buffer source")) {
    return false;
  }

  AVBufferSrcParameters *buffersrc_params = av_buffersrc_parameters_alloc();
  if (!buffersrc_params) {
    std::cerr << "Failed to allocate AVBufferSrcParameters." << std::endl;
    return false;
  }

  buffersrc_params->hw_frames_ctx = av_buffer_ref(hw_frames_ctx);
  if (!buffersrc_params->hw_frames_ctx) {
    std::cerr << "Failed to ref manually allocated hw_frames_ctx for "
                 "buffersrc_params."
              << std::endl;
    av_free(buffersrc_params);
    return false;
  }

  ret = av_buffersrc_parameters_set(buffersrc_ctx, buffersrc_params);
  if (check_error(ret, "Failed to set parameters on buffersrc")) {
    av_free(buffersrc_params);
    return false;
  }
  av_free(buffersrc_params);

  ret = avfilter_graph_create_filter(&buffersink_ctx, buffersink, "out",
                                     nullptr, nullptr, filter_graph);
  if (check_error(ret, "Cannot create buffer sink")) {
    return false;
  }

  enum AVPixelFormat pix_fmts[] = {AV_PIX_FMT_NONE};
  ret = av_opt_set_int_list(buffersink_ctx, "pix_fmts", pix_fmts,
                            AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN);
  if (check_error(ret, "Cannot set output pixel format")) {
    return false;
  }

  AVFilterInOut *outputs = avfilter_inout_alloc();
  AVFilterInOut *inputs = avfilter_inout_alloc();

  if (!outputs || !inputs) {
    std::cerr << "Failed to allocate AVFilterInOut structs." << std::endl;
    avfilter_inout_free(&outputs);
    avfilter_inout_free(&inputs);
    return false;
  }

  outputs->name = av_strdup("in");
  outputs->filter_ctx = buffersrc_ctx;
  outputs->pad_idx = 0;
  outputs->next = nullptr;

  inputs->name = av_strdup("out");
  inputs->filter_ctx = buffersink_ctx;
  inputs->pad_idx = 0;
  inputs->next = nullptr;

  // Use the member variable filter_descr_ here
  ret = avfilter_graph_parse_ptr(filter_graph, filter_descr_.c_str(), &inputs,
                                 &outputs, nullptr);
  if (check_error(ret, "Cannot parse filter graph")) {
    avfilter_inout_free(&outputs);
    avfilter_inout_free(&inputs);
    return false;
  }

  std::cout << "Configuring filter graph..." << std::endl;
  ret = avfilter_graph_config(filter_graph, nullptr);
  if (check_error(ret, "Failed to configure filter graph")) {
    return false;
  }

  frame_width_ = buffersink_ctx->inputs[0]->w;
  frame_height_ = buffersink_ctx->inputs[0]->h;

  return true;
}

void FFMPEGVideo::cleanup() {
  std::cout << "Cleaning up FFmpeg resources..." << std::endl;
  avfilter_graph_free(&filter_graph);
  avcodec_free_context(&dec_ctx);
  avformat_close_input(&fmt_ctx);
  av_packet_free(&pkt);
  av_frame_free(&frame);
  av_frame_free(&filt_frame);
  av_buffer_unref(&hw_frames_ctx);
  av_buffer_unref(&hw_device_ctx);
  std::cout << "FFmpeg resources cleaned up." << std::endl;
}
