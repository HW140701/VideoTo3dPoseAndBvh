# Video to 3DPose and Bvh motion file

<p align="center"><img src="asset/wujing.gif" width="100%" alt="" /></p>
<p align="center"><img src="asset/kunkun_alphapose.gif" width="100%" alt="" /></p>
<p align="center"><img src="asset/example_bvh.gif" width="100%" alt="" /></p>
<p align="center"><img src="asset/example_bvh_1.gif" width="100%" alt="" /></p>

This project integrates some project working, example as [VideoPose3D](https://github.com/facebookresearch/VideoPose3D),[video-to-pose3D](https://github.com/zh-plus/video-to-pose3D) , [video2bvh](https://github.com/KevinLTT/video2bvh "video2bvh"), [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose), [Higher-HRNet-Human-Pose-Estimation](https://github.com/HRNet/Higher-HRNet-Human-Pose-Estimation),[openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), thanks for the mentioned above project.

The project extracted the 2d joint key point from the video by using [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose),[HRNet](https://github.com/HRNet/Higher-HRNet-Human-Pose-Estimation) and so on. Then transform the 2d point to the 3D joint point by using [VideoPose3D](https://github.com/facebookresearch/VideoPose3D). Finally We convert the 3d joint point to the bvh motion file.

# Environment
- Windows 10
- Anaconda
- Python > 3.6

# Dependencies
>You can refer to the project dependencies of [video-to-pose3D](https://github.com/zh-plus/video-to-pose3D) for setting.

This is the dependencies of the project of [video-to-pose3D](https://github.com/zh-plus/video-to-pose3D), and modifyed by me to solve some bug.

- **Packages**
  - Pytorch > 1.1.0 (I use the Pytorch1.1.0 - GPU)
  - [torchsample](https://github.com/MVIG-SJTU/AlphaPose/issues/71#issuecomment-398616495)
  - [ffmpeg](https://ffmpeg.org/download.html) (note:you must copy the ffmpeg.exe to the directory of python install)
  - tqdm
  - pillow
  - scipy
  - pandas
  - h5py
  - visdom
  - nibabel
  - opencv-python (install with pip)
  - matplotlib
- **2D Joint detectors**
     - Alphapose (Recommended)
       - Download **duc_se.pth** from ([Google Drive](https://drive.google.com/open?id=1OPORTWB2cwd5YTVBX-NE8fsauZJWsrtW) | [Baidu pan](https://pan.baidu.com/s/15jbRNKuslzm5wRSgUVytrA)),
         place to `./joints_detectors/Alphapose/models/sppe`
       - Download **yolov3-spp.weights** from ([Google Drive](https://drive.google.com/open?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC) | [Baidu pan](https://pan.baidu.com/s/1Zb2REEIk8tcahDa8KacPNA)),
         place to `./joints_detectors/Alphapose/models/yolo`
     - HR-Net (Bad 3d joints performance in my testing environment)
       - Download **pose_hrnet*** from [Google Drive](https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA), 
         place to `./joints_detectors/hrnet/models/pytorch/pose_coco/`
       - Download **yolov3.weights** from [here](https://pjreddie.com/media/files/yolov3.weights),
         place to `./joints_detectors/hrnet/lib/detector/yolo`
- **3D Joint detectors**
      - Download **pretrained_h36m_detectron_coco.bin** from [here](https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin),
        place it into `./checkpoint` folder
- **2D Pose trackers (Optional)**
      - PoseFlow (Recommended)
        No extra dependences
      - LightTrack (Bad 2d tracking performance in my testing environment)
        - See [original README](https://github.com/Guanghan/lighttrack), and perform same *get started step* on `./pose_trackers/lighttrack`

# How to Use it
Please place your video to the .\outputs\inputvideo, and setting the path to the videopose.py, like this 

    inference_video('outputs/inputvideo/kunkun_cut.mp4', 'alpha_pose')

Waiting some minute, you can see the output video in the \outputs\outputvideo directory,and the bvh file in the \outputs\outputvideo\alpha_pose_kunkun_cut\bvh directory.
