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
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

// Helper function to get FFmpeg error strings
std::string av_error_to_string(int errnum) {
  char err_buf[AV_ERROR_MAX_STRING_SIZE];
  av_strerror(errnum, err_buf, sizeof(err_buf));
  return std::string(err_buf);
}

// Function to handle FFmpeg errors
static bool check_error(int ret, const std::string &msg) {
  if (ret < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(ret, errbuf, sizeof(errbuf));
    std::cerr << msg << ": " << errbuf << std::endl;
    return true; // Indicate error
  }
  return false; // Indicate success
}

class FFMPEGVideo {
public:
  // Constructor: Initializes FFmpeg and hardware components
  FFMPEGVideo(const std::string &filename)
      : input_filename_(filename), fmt_ctx(nullptr), dec_ctx(nullptr),
        filter_graph(nullptr), buffersrc_ctx(nullptr), buffersink_ctx(nullptr),
        hw_device_ctx(nullptr), hw_frames_ctx(nullptr), pkt(nullptr),
        frame(nullptr), filt_frame(nullptr), video_stream_idx(-1),
        initialized(false), frame_count_(0), // Initialize frame counter
        total_frames_(0),                    // Initialize total frames
        frame_width_(0),                     // Initialize output frame width
        frame_height_(0),                    // Initialize output frame height
        video_time_base_({0, 1}),            // Initialize time base
        current_frame_pts_(AV_NOPTS_VALUE),  // Initialize current PTS
        current_frame_time_seconds_(0.0)     // Initialize current time
  {
    pkt = av_packet_alloc();
    frame = av_frame_alloc();
    filt_frame = av_frame_alloc();

    if (!pkt || !frame || !filt_frame) {
      std::cerr << "Failed to allocate AVPacket or AVFrame. Out of memory?"
                << std::endl;
      return; // Initialization failed
    }

    initialized = init();
  }

  // Destructor: Cleans up all FFmpeg resources
  ~FFMPEGVideo() { cleanup(); }

  // Check if the FFmpeg pipeline was successfully initialized
  bool isInitialized() const { return initialized; }

  // Fetches the next processed video frame
  // Returns true if a frame is available, false if EOF or error
  bool GetNextFrame(cv::Mat &output_mat) {
    if (!initialized) {
      std::cerr << "FFMPEGVideo not initialized. Cannot get frame."
                << std::endl;
      return false;
    }

    int ret = 0;
    bool frame_retrieved =
        false; // Flag to indicate if a frame has been successfully retrieved
    bool end_of_input_reached =
        false; // Flag to indicate if all input packets have been read

    // Loop as long as we haven't found a frame AND we still have potential
    // input/flushing
    while (!frame_retrieved) {
      // --- Phase 1: Try to pull a filtered frame from the buffersink ---
      // This is the primary goal: get an output frame.
      ret = av_buffersink_get_frame(buffersink_ctx, filt_frame);
      if (ret >= 0) {
        // Successfully got a filtered frame. Process and return true.
        return process_retrieved_frame(output_mat);
      } else if (ret == AVERROR(EAGAIN)) {
        // Filter graph needs more input. Proceed to decoding/reading.
      } else if (ret == AVERROR_EOF) {
        // Filter graph is completely flushed and empty. No more frames can come
        // from it. This implies all input has been processed and drained.
        end_of_input_reached = true;
        // Break out of the primary loop to enter the final flushing sequence if
        // needed.
        break;
      } else {
        // Unrecoverable error from buffersink.
        check_error(ret, "Error receiving frame from filter graph");
        return false;
      }

      // --- Phase 2: If no filtered frame, try to decode more frames and feed
      // the filter graph --- Only attempt this if we haven't reached the end of
      // the input stream.
      if (!end_of_input_reached) {
        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret >= 0) {
          // Successfully got a decoded frame.
          frame->pts = frame->best_effort_timestamp; // Set PTS for consistent
                                                     // timestamping.

          // Feed the decoded frame into the filter graph.
          int add_frame_ret = av_buffersrc_add_frame_flags(
              buffersrc_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF);
          if (check_error(add_frame_ret,
                          "Error feeding frame to filter graph")) {
            av_frame_unref(frame);
            return false;
          }
          av_frame_unref(frame); // Frame consumed by buffersrc.
          // Continue the loop to immediately try and pull from buffersink
          // again, as feeding might have made a filtered frame available.
          continue;
        } else if (ret == AVERROR(EAGAIN)) {
          // Decoder needs more packets. Proceed to reading input.
        } else if (ret == AVERROR_EOF) {
          // Decoder is fully flushed. This means we've processed all packets
          // that were sent to it. We must now enter the overall flushing phase.
          end_of_input_reached = true;
          // Break out of the primary loop to enter the final flushing sequence.
          break;
        } else {
          // Unrecoverable error from decoder.
          check_error(ret, "Error receiving frame from decoder");
          return false;
        }

        // --- Phase 3: If no decoded frames, read more raw packets from input
        // file --- Only attempt this if we still expect more input.
        ret = av_read_frame(fmt_ctx, pkt);
        if (ret < 0) { // Error or EOF
          if (ret == AVERROR_EOF) {
            end_of_input_reached = true; // All input packets read.
            // Break out of the primary loop to enter the final flushing
            // sequence.
            break;
          } else {
            // Unrecoverable error reading input.
            check_error(ret, "Error reading packet from input");
            return false;
          }
        }

        if (pkt->stream_index == video_stream_idx) {
          // Send the packet to the decoder.
          // This inner loop handles AVERROR(EAGAIN) by draining the decoder
          // until it accepts the packet.
          int send_packet_ret;
          while ((send_packet_ret = avcodec_send_packet(dec_ctx, pkt)) ==
                 AVERROR(EAGAIN)) {
            // Decoder buffer is full. Try to drain it by receiving frames.
            int drain_ret = avcodec_receive_frame(dec_ctx, frame);
            if (drain_ret >= 0) {
              // Successfully drained a frame. Feed it to filter graph.
              int add_drain_frame_to_filter_ret = av_buffersrc_add_frame_flags(
                  buffersrc_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF);
              if (check_error(add_drain_frame_to_filter_ret,
                              "Error feeding drained frame to filter graph")) {
                av_frame_unref(frame);
                av_packet_unref(pkt);
                return false;
              }
              av_frame_unref(frame); // Frame consumed by buffersrc.

              // After draining and feeding, immediately check if a filtered
              // frame is available. If so, we can return it.
              int pull_filtered_after_drain_ret =
                  av_buffersink_get_frame(buffersink_ctx, filt_frame);
              if (pull_filtered_after_drain_ret >= 0) {
                av_packet_unref(pkt); // Packet processed.
                return process_retrieved_frame(
                    output_mat); // Frame found, exit completely.
              } else if (pull_filtered_after_drain_ret != AVERROR(EAGAIN) &&
                         pull_filtered_after_drain_ret != AVERROR_EOF) {
                check_error(pull_filtered_after_drain_ret,
                            "Error receiving filtered frame after draining in "
                            "send retry");
                av_packet_unref(pkt);
                return false;
              }
              // If still EAGAIN, the loop continues trying to send the packet.
            } else if (drain_ret == AVERROR(EAGAIN) ||
                       drain_ret == AVERROR_EOF) {
              // Cannot drain decoder further now. Break from send packet retry
              // loop. This case means the decoder is stalled and needs more
              // input, or it's fully flushed.
              std::cerr << "Warning: Cannot drain decoder to send packet. "
                           "Breaking send retry loop."
                        << std::endl;
              break;
            } else {
              check_error(drain_ret, "Error receiving frame from decoder "
                                     "during packet send retry");
              av_packet_unref(pkt);
              return false;
            }
          } // End of while(avcodec_send_packet == AVERROR(EAGAIN))

          if (send_packet_ret <
              0) { // If sending the packet ultimately failed (not EAGAIN)
            check_error(send_packet_ret,
                        "Error sending packet to decoder after retries");
            av_packet_unref(pkt);
            return false;
          }
        }
        av_packet_unref(pkt); // Packet has been sent to decoder, unref it.
        // Continue the main loop to try and get a frame from the pipeline.
      } // End of if (!end_of_input_reached) for reading packets
    } // End of main while(!frame_retrieved) loop

    // --- Final Flushing Logic (executed when end_of_input_reached is true) ---
    // This block ensures all remaining frames in the decoder and filter graph
    // are pulled.
    if (end_of_input_reached) {
      std::cout << "Initiating pipeline flushing..." << std::endl;

      // 1. Send NULL packet to decoder to signal end of stream and put it in
      // draining mode.
      avcodec_send_packet(dec_ctx, nullptr);

      // 2. Receive all remaining decoded frames from the decoder and feed them
      // to the filter graph.
      while (true) {
        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
          break; // Decoder fully flushed.
        } else if (check_error(ret,
                               "Error receiving flushed frame from decoder")) {
          return false;
        }
        frame->pts = frame->best_effort_timestamp; // Ensure PTS is set for
                                                   // flushed frames.
        int add_flush_frame_to_filter_ret = av_buffersrc_add_frame_flags(
            buffersrc_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF);
        if (check_error(add_flush_frame_to_filter_ret,
                        "Error feeding flushed frame to filter graph")) {
          av_frame_unref(frame);
          return false;
        }
        av_frame_unref(frame);
      }

      // 3. Send NULL frame to buffersrc to signal end of input for the filter
      // graph. This puts the filter graph in draining mode.
      int add_flush_to_buffersrc_ret =
          av_buffersrc_add_frame_flags(buffersrc_ctx, nullptr, 0);
      if (check_error(add_flush_to_buffersrc_ret,
                      "Error flushing buffer source")) {
        return false;
      }

      // 4. Receive all remaining filtered frames from the buffersink.
      while (true) {
        ret = av_buffersink_get_frame(buffersink_ctx, filt_frame);
        if (ret >= 0) {
          // Found a flushed frame. Process and return.
          return process_retrieved_frame(output_mat);
        } else if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
          break; // Filter graph fully flushed, no more frames.
        } else {
          check_error(ret, "Error receiving flushed frame from filter graph");
          return false;
        }
      }
    }
    return false; // No more frames from pipeline after full flush.
  }

  // New getter methods
  int get_frame_width() const { return frame_width_; }

  int get_frame_height() const { return frame_height_; }

  int get_frame_id() const { return frame_count_; }

  // Get total frames (may be an approximation or 0 depending on container
  // format)
  int get_frame_total() const { return total_frames_; }

  // Get PTS of the last retrieved frame
  int64_t get_last_frame_pts() const { return current_frame_pts_; }

  // Get time in seconds of the last retrieved frame's PTS
  double get_last_frame_time_seconds() const {
    return current_frame_time_seconds_;
  }

private:
  std::string input_filename_;
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

  int frame_count_;  // Counter for frames processed so far (starting from 1)
  int total_frames_; // Total frames in the video (approximation)
  int frame_width_;  // Width of the output frames
  int frame_height_; // Height of the output frames
  AVRational video_time_base_; // Time base of the video stream

  int64_t current_frame_pts_; // Stores PTS of the most recently retrieved frame
  double current_frame_time_seconds_; // Stores time in seconds of the most
                                      // recently retrieved frame

  // Private helper function to handle a successfully retrieved filtered frame.
  // Encapsulates common logic previously under the 'handle_frame' label.
  bool process_retrieved_frame(cv::Mat &output_mat_ref) {

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
      current_frame_time_seconds_ =
          current_frame_pts_ * av_q2d(video_time_base_);
    } else {
      current_frame_time_seconds_ = 0.0;
    }

    av_frame_unref(filt_frame);
    return true;
  }

  // Callback function to select the hardware pixel format for the decoder
  static enum AVPixelFormat get_hw_format(AVCodecContext *ctx,
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
        std::cout << "Negotiating HW Pixel Format: AV_PIX_FMT_NV12 for decoder "
                     "output."
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
  bool init() {
    int ret = 0;

    // --- 1. Open input file and find stream info ---
    std::cout << "Opening input file: " << input_filename_ << std::endl;
    ret = avformat_open_input(&fmt_ctx, input_filename_.c_str(), nullptr,
                              nullptr);
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
      // Approximate total frames from duration and average frame rate
      double duration_seconds =
          (double)fmt_ctx->streams[video_stream_idx]->duration *
          av_q2d(video_time_base_);
      double fps_double =
          av_q2d(fmt_ctx->streams[video_stream_idx]->avg_frame_rate);
      total_frames_ = static_cast<int>(duration_seconds * fps_double);
      if (total_frames_ == 0 &&
          fmt_ctx->streams[video_stream_idx]->duration > 0) {
        // If duration-based calculation also yields 0, it means it's unreliable
        std::cerr << "Total frames calculated as 0 despite positive duration."
                  << std::endl;
      }
    } else {
      std::cerr << "Warning: Total frames count is unavailable or unreliable "
                   "for this stream."
                << std::endl;
    }

    // --- 2. Initialize Hardware Acceleration ---
    AVHWDeviceType hw_type = AV_HWDEVICE_TYPE_NONE;
    const char *hw_device_type_name =
        "rkmpp"; // Rockchip Media Processing Platform

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

    // Create HW device context
    ret = av_hwdevice_ctx_create(&hw_device_ctx, hw_type, nullptr, nullptr, 0);
    if (check_error(ret, "Failed to create HW device context")) {
      return false;
    }
    std::cout << "Successfully created HW device context: "
              << hw_device_type_name << std::endl;

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
    if (check_error(ret,
                    "Failed to copy codec parameters to decoder context")) {
      return false;
    }

    // Assign the HW device context to the decoder context.
    // This is necessary for the decoder to utilize the hardware device.
    dec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    if (!dec_ctx->hw_device_ctx) {
      std::cerr << "Failed to set HW device context for codec context."
                << std::endl;
      return false;
    }
    std::cout << "Hardware device context set for codec context." << std::endl;

    // Set the callback function to negotiate the hardware pixel format for
    // decoder output.
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
    // This explicitly tells the decoder what kind of hardware frames it should
    // output, and how they should be pooled.
    hw_frames_ctx = av_hwframe_ctx_alloc(hw_device_ctx);
    if (!hw_frames_ctx) {
      std::cerr << "Failed to allocate AVHWFramesContext." << std::endl;
      return false;
    }

    AVHWFramesContext *frames_ctx_data =
        (AVHWFramesContext *)(hw_frames_ctx->data);
    frames_ctx_data->format =
        dec_ctx->pix_fmt; // This should be the HW pixel format chosen by
                          // get_hw_format (e.g., DRM_PRIME).
    frames_ctx_data->sw_format =
        AV_PIX_FMT_NV12; // The software format that corresponds to the HW
                         // format (for mapping).
    frames_ctx_data->width = dec_ctx->width;
    frames_ctx_data->height = dec_ctx->height;
    frames_ctx_data->initial_pool_size =
        0; // FFmpeg will manage pool size dynamically.

    ret = av_hwframe_ctx_init(hw_frames_ctx);
    if (check_error(ret, "Failed to initialize AVHWFramesContext")) {
      return false;
    }
    std::cout << "Successfully initialized AVHWFramesContext for decoder's "
                 "output (format: "
              << av_get_pix_fmt_name(frames_ctx_data->format) << ")."
              << std::endl;

    // Assign the manually initialized hw_frames_ctx to the decoder.
    // The decoder will use this for allocating hardware-backed frames.
    av_buffer_unref(&dec_ctx->hw_frames_ctx); // Release any existing ref
                                              // (should be null initially).
    dec_ctx->hw_frames_ctx = av_buffer_ref(hw_frames_ctx);
    if (!dec_ctx->hw_frames_ctx) {
      std::cerr << "Failed to assign allocated hw_frames_ctx to decoder "
                   "context after init."
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
    // Arguments for the buffer source filter. It defines the properties of the
    // input to the filter graph.
    std::string buffersrc_args =
        "video_size=" + std::to_string(dec_ctx->width) + "x" +
        std::to_string(dec_ctx->height) + ":pix_fmt=" +
        av_get_pix_fmt_name(
            dec_ctx->pix_fmt) + // Use decoder's output pix_fmt (HW format)
        ":time_base=" +
        std::to_string(time_base.num) + "/" + std::to_string(time_base.den) +
        ":pixel_aspect=" + std::to_string(dec_ctx->sample_aspect_ratio.num) +
        "/" + std::to_string(dec_ctx->sample_aspect_ratio.den);

    // Filter description string:
    // 1. scale_rkrga: Hardware-accelerated scaling using Rockchip RGA.
    //    w=1280:h=720: Rescale to 1280x720.
    //    format=bgr24: Output pixel format from RGA is BGR24.
    // 2. hwmap=mode=read: Maps the hardware frame to a software-accessible
    // frame for reading.
    //    This is crucial if the filter graph's last output is a hardware frame
    //    and OpenCV needs CPU access.
    // 3. format=bgr24: Final software pixel format before passing to OpenCV.
    std::string filter_descr =
        "scale_rkrga=w=1280:h=720:format=bgr24,hwmap=mode=read,format=bgr24";

    const AVFilter *buffersrc =
        avfilter_get_by_name("buffer"); // Input filter to the graph.
    const AVFilter *buffersink =
        avfilter_get_by_name("buffersink"); // Output filter from the graph.

    // Create the buffer source filter.
    ret = avfilter_graph_create_filter(&buffersrc_ctx, buffersrc, "in",
                                       buffersrc_args.c_str(), nullptr,
                                       filter_graph);
    if (check_error(ret, "Cannot create buffer source")) {
      return false;
    }

    // Set the hardware frames context for the buffer source.
    // This tells the buffer source that it will receive hardware-accelerated
    // frames.
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

    // Set colorspace and color_range directly from decoder context
    buffersrc_params->color_space = dec_ctx->colorspace;
    buffersrc_params->color_range = dec_ctx->color_range;

    ret = av_buffersrc_parameters_set(buffersrc_ctx, buffersrc_params);
    if (check_error(ret, "Failed to set parameters on buffersrc")) {
      av_free(buffersrc_params);
      return false;
    }
    av_free(buffersrc_params); // Parameters struct is copied, so free it.

    // Create the buffer sink filter.
    ret = avfilter_graph_create_filter(&buffersink_ctx, buffersink, "out",
                                       nullptr, nullptr, filter_graph);
    if (check_error(ret, "Cannot create buffer sink")) {
      return false;
    }

    // Set the desired output pixel formats for the buffer sink.
    // We want BGR24 for direct compatibility with OpenCV.
    enum AVPixelFormat pix_fmts[] = {AV_PIX_FMT_NONE};
    ret = av_opt_set_int_list(buffersink_ctx, "pix_fmts", pix_fmts,
                              AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN);
    if (check_error(ret, "Cannot set output pixel format")) {
      return false;
    }

    // Setup input/output pads for parsing the filter graph string.
    AVFilterInOut *outputs = avfilter_inout_alloc();
    AVFilterInOut *inputs = avfilter_inout_alloc();

    if (!outputs || !inputs) {
      std::cerr << "Failed to allocate AVFilterInOut structs." << std::endl;
      avfilter_inout_free(
          &outputs); // Ensure allocated structs are freed on error.
      avfilter_inout_free(&inputs);
      return false;
    }

    // Link the buffer source output to the first filter's input.
    outputs->name = av_strdup("in");
    outputs->filter_ctx = buffersrc_ctx;
    outputs->pad_idx = 0;
    outputs->next = nullptr;

    // Link the last filter's output to the buffer sink input.
    inputs->name = av_strdup("out");
    inputs->filter_ctx = buffersink_ctx;
    inputs->pad_idx = 0;
    inputs->next = nullptr;

    // Parse the filter graph string and link the filters.
    ret = avfilter_graph_parse_ptr(filter_graph, filter_descr.c_str(), &inputs,
                                   &outputs, nullptr);
    if (check_error(ret, "Cannot parse filter graph")) {
      avfilter_inout_free(&outputs); // Free on error.
      avfilter_inout_free(&inputs);
      return false;
    }
    // outputs and inputs are freed by avfilter_graph_parse_ptr on success.

    std::cout << "Configuring filter graph..." << std::endl;
    // Configure the filter graph. This step connects all filters and
    // initializes their contexts.
    ret = avfilter_graph_config(filter_graph, nullptr);
    if (check_error(ret, "Failed to configure filter graph")) {
      return false;
    }

    // Set initial frame_width_ and frame_height_ from filter graph.
    // These are the expected output dimensions after filtering.
    frame_width_ = buffersink_ctx->inputs[0]->w;
    frame_height_ = buffersink_ctx->inputs[0]->h;

    return true; // Initialization successful
  }

  // Cleans up all FFmpeg components
  void cleanup() {
    std::cout << "Cleaning up FFmpeg resources..." << std::endl;
    avfilter_graph_free(
        &filter_graph);              // Frees filter_graph and all its filters.
    avcodec_free_context(&dec_ctx);  // Frees the decoder context.
    avformat_close_input(&fmt_ctx);  // Closes the input file and frees fmt_ctx.
    av_packet_free(&pkt);            // Frees the packet.
    av_frame_free(&frame);           // Frees the raw frame.
    av_frame_free(&filt_frame);      // Frees the filtered frame.
    av_buffer_unref(&hw_frames_ctx); // Decrements ref count for hw_frames_ctx.
    av_buffer_unref(&hw_device_ctx); // Decrements ref count for hw_device_ctx.
    std::cout << "FFmpeg resources cleaned up." << std::endl;
  }
};

int main(int argc, char **argv) {
  // Default video file path.
  // Replace with a valid path to an HEVC video file that rkmpp can decode.
  const char *input_filename = "/data/video/1/2025/06/24/H121643.asf";
  if (argc > 1) {
    input_filename = argv[1];
  }

  // Create an instance of FFMPEGVideo processor.
  FFMPEGVideo video_processor(input_filename);

  if (!video_processor.isInitialized()) {
    std::cerr << "Failed to initialize FFMPEGVideo processor. Exiting."
              << std::endl;
    return 1;
  }

  // Setup OpenCV window for display.
  cv::namedWindow("Video Playback", cv::WINDOW_AUTOSIZE);
  cv::Mat frame_mat; // Mat to hold the output frame.

  std::cout << "Starting video playback loop..." << std::endl;

  // Main playback loop: continuously get frames and display them.
  while (video_processor.GetNextFrame(frame_mat)) {
    // Print frame information.
    std::cout << "Displaying frame " << video_processor.get_frame_id() << " of "
              << (video_processor.get_frame_total() > 0
                      ? std::to_string(video_processor.get_frame_total())
                      : "unknown total")
              << " (Resolution: " << video_processor.get_frame_width() << "x"
              << video_processor.get_frame_height()
              << " , Channels: " << frame_mat.channels()
              << ", PTS: " << video_processor.get_last_frame_pts()
              << " Time: " << std::fixed << std::setprecision(4)
              << video_processor.get_last_frame_time_seconds() << "s)"
              << std::endl;

    if ((video_processor.get_frame_id() % 100) == 0) {
      // Display the frame using OpenCV.
      cv::imshow("Video Playback", frame_mat);
      char key = cv::waitKey(0);     // Wait for key press (allows GUI events)
      if (key == 'q' || key == 27) { // 'q' key or ESC key to quit.
        std::cout << "User requested exit." << std::endl;
        break;
      }
    }
  }

  std::cout << "Video playback finished." << std::endl;
  cv::destroyAllWindows(); // Close OpenCV windows.

  return 0;
}
