import cv2
import ffmpeg_video
import numpy as np

# Define the path to your video file
# Make sure this video file exists and is accessible
VIDEO_FILE = "/data/video/1/2025/06/24/H121643.asf"

# Define a custom filter description
# Example 1: Rescale to 640x360 and apply a grayscale filter
# CUSTOM_FILTER_DESCR = "scale_rkrga=w=640:h=360:format=bgr24,hwmap=mode=read,format=gray"
# Note: if using grayscale (format=gray), your mat_to_numpy might need adjustment for CV_8UC1
# For now, let's stick to BGR24 output for simplicity with OpenCV display.

# Example 2: Rescale to 640x360 but keep BGR24
CUSTOM_FILTER_DESCR = "scale_rkrga=w=640:h=360:format=bgr24,hwmap=mode=read,format=bgr24"

# Example 3: Original filter for 1280x720
# CUSTOM_FILTER_DESCR = "scale_rkrga=w=1280:h=720:format=bgr24,hwmap=mode=read,format=bgr24"


def main():
    print(f"Attempting to initialize FFMPEGVideo with: {VIDEO_FILE}")
    print(f"Using filter: {CUSTOM_FILTER_DESCR}")
    try:
        # Create an instance of the FFMPEGVideo class, passing the custom filter
        processor = ffmpeg_video.FFMPEGVideo(VIDEO_FILE, CUSTOM_FILTER_DESCR)
        # If you wanted to use the default filter, you'd just do:
        # processor = ffmpeg_video.FFMPEGVideo(VIDEO_FILE)


        if not processor.is_initialized():
            print("Failed to initialize FFMPEGVideo processor. Exiting.")
            return

        print("\n--- Video Processor Initialized ---")
        print(f"Output Frame Resolution: {processor.get_frame_width()}x{processor.get_frame_height()}")
        print(f"Estimated Total Frames: {processor.get_frame_total()}")
        print("-----------------------------------\n")

        # Create a window for displaying frames
        cv2.namedWindow("Video Playback (Python)", cv2.WINDOW_AUTOSIZE)

        frame_num = 0
        while True:
            # Get the next frame as a NumPy array
            np_frame = processor.get_next_frame()

            if np_frame is None:
                print("End of stream or error, no more frames.")
                break

            frame_num += 1
            # Print frame details
            print(f"Displaying frame {processor.get_frame_id()} "
                  f"(Actual count: {frame_num}) "
                  f"(Resolution: {np_frame.shape[1]}x{np_frame.shape[0]}, "
                  f"PTS: {processor.get_last_frame_pts()}, "
                  f"Time: {processor.get_last_frame_time_seconds():.4f}s)")

            # Display the frame using OpenCV (NumPy array is compatible)
            cv2.imshow("Video Playback (Python)", np_frame)

            # Wait for 1ms and check for key press
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27: # 'q' key or ESC key
                print("User requested exit.")
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Video playback finished.")

if __name__ == "__main__":
    main()
