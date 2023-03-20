
req : numpy,
      twitchstream, //just main
      open cv ie cv2
      Pillow, PIL
      cudatoolkit but need conda (maybe not) for that so idk. install that https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local
                                                              and update the path in CellularAutomaton


git push -u origin main


cudatoolkit : C:\Users\lele3\AppData\Local\Temp\CUDA


do not use the video generator unless you wanna fill your hard drive, export the images and process them in a dedicated software (ex : da vinci)

main is not main. run cellular automata after changing the parameters
might crash if there is no temp and Save folder (assuming you didn't change those paths in the config)

I would not advice changing _THREADSPERBLOCK and _BLOCKSPERGRID unless you know what you're doing (and if ou don't, keep the multiples of 2 i guess? not too high as well; ressources : https://youtu.be/9bBsvpg-Xlk)




pbs : Y IT FLIP WTF // fixed. idiot put - instead of +
      borders fcky? why?






















#
