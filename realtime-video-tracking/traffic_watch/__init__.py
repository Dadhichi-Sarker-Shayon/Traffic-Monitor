"""traffic_watch — real-time traffic observer built around OpenPSG.

Pipeline:
    capture (file/webcam/RTSP)
      -> detector + tracker (YOLOv8 + ByteTrack, vehicles/persons)
      -> scene layer (OpenPSG PSGTR scene graphs, optional/interleaved)
      -> event engine (jam / jam origin / accident / stopped / conflict)
      -> overlay renderer (everything marked on the video)
"""

__version__ = "0.1.0"
