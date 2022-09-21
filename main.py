from __future__ import print_function
from twitchstream.outputvideo import TwitchBufferedOutputStream
import argparse
import numpy as np
import CellularAutomaton as ca


size = (256, 256)
FPS = 1
Buffer = 10

"""
main.py -s STREAMKEYHERE
"""





#https://317070.github.io/python/




frame_colorShift = [110/255, 220/255, 255/255]


def MakeFrame(input):
    output = np.zeros((input.shape[0], input.shape[1], 3))
    for x in range(output.shape[0]):
        for y in range(output.shape[1]):
            val = input[x][y]
            #if val != 0: print(val)
            output[x][y] = [val * frame_colorShift[0],val * frame_colorShift[1],val * frame_colorShift[2]]
    return output





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-s', '--streamkey',help='twitch streamkey', required=True)
    args = parser.parse_args()


    videostream = TwitchBufferedOutputStream(twitch_stream_key=args.streamkey, width=size[0], height=size[1], fps=FPS, verbose=True)

    frame = np.zeros((size[0], size[1]))
    frame[int(size[0]/2)][int(size[1]/2)] = 1 #starting grid
    filter = ca.NCA_Filter #filter to apply
    function = ca.f #activation function
    while True:
        if videostream.get_video_frame_buffer_state() < Buffer:

            frame = ca.NextStep(frame, filter, function)
            StreamFrame = MakeFrame(frame)
            videostream.send_video_frame(StreamFrame)





















#
