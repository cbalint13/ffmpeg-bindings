# ffmpeg-bindings
High performance FFMPEG video reader bindings (with HW acceleration)

These bindings **accelerates** video **decoding** & **scaling** within CV / ML apps around OpenCV or numpy.

---

## Overview
* Implementation of ```C++``` & ```python``` FFMPEG reader bindings using HW acceleration (via ```mpp``` driver).
* It also rescales and converts to *BGR24/GRAY8* the video frames using HW acceleration (via ```rga``` driver).

### Python binding example

```python
import ffmpeg_video

vid_filter = "scale_rkrga=w=640:h=360:format=bgr24,hwmap=mode=read,format=bgr24"
cap = ffmpeg_video.FFMPEGVideo("my_video.mp4", vid_filter)

while True:
  np_frame = cap.get_next_frame()

  print(f"Displaying frame {cap.get_frame_id()} "
        f"(Actual count: {frame_num}) "
        f"(Resolution: {np_frame.shape}, "
        f"PTS: {cap.get_last_frame_pts()}, "
        f"Time: {cap.get_last_frame_time_seconds():.4f}s)")

  cv2.imshow("Video Playback (Python)", np_frame)
  key = cv2.waitKey(0)
```

## Building
* This use custom (rockchip) ffmpeg branch: https://github.com/nyanmisaka/ffmpeg-rockchip/tree/7.1
* See advanced usage with dedicated hardware processing: https://github.com/nyanmisaka/ffmpeg-rockchip/wiki
* It can be adapted to any other (vanilla avdecode/avfilter) HW accelators or non accelerated scenarios.

There are readily usable .rpm packages for Fedora:
* ```libmpp-rockchip```, ```librga-rockchip```, ```ffmpeg```
* Via repository: https://copr.fedorainfracloud.org/coprs/rezso/ML


## Samples
* Examples provided here are in ```C++``` and ```python``` (via binding).
* The equivalent HW accelerated CLI invokation of ffmpeg would be:
  ```
  ffmpeg -fflags +genpts -hwaccel rkmpp -hwaccel_output_format drm_prime \
         -vcodec hevc_rkmpp -afbc rga \
         -i /data/video/1/2025/06/27/H124841.asf \
         -vf hwupload,scale_rkrga=w=1280:h=720:format=bgr24,hwmap=mode=read,format=bgr24 \
         -f rawvideo pipe:
  ```
* This yields > 500 FPS on a busy rk3588 SoC:
  ```
  Input #0, asf, from '/data/video/1/2025/06/27/H124841.asf':
    Duration: 00:02:59.95, start: -0.014000, bitrate: 1124 kb/s
    Stream #0:0: Video: hevc (Main) (HEVC / 0x43564548), yuvj420p(pc, bt709),
                        2880x1616, 5 fps, 5 tbr, 1k tbn
  Stream mapping:
    Stream #0:0 -> #0:0 (hevc (hevc_rkmpp) -> wrapped_avframe (native))
  rga_api version 1.10.4_[1]
  Output #0, null, to '/dev/null':
    Metadata:
      encoder         : Lavf61.7.100
    Stream #0:0: Video: wrapped_avframe, bgr24(pc, gbr/bt709/bt709, progressive),
                        1280x718, q=2-31, 200 kb/s, 5 fps, 5 tbn
        Metadata:
          encoder         : Lavc61.19.101 wrapped_avframe
  [out#0/null @ 0x55b25accc0] video:386KiB audio:0KiB subtitle:0KiB
                              other streams:0KiB global headers:0KiB
                              muxing overhead: unknown
  frame=  899 fps=528 q=-0.0 Lsize=N/A time=00:02:59.80 bitrate=N/A speed=75.7x    
  ```
