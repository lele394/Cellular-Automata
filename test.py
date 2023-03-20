

x = 0
y = 0


_filter = [[-0.99038735,  0.82367216, -0.99038735],
            [ 0.82367216,  0.31695098,  0.82367216],
            [-0.99038735,  0.82367216, -0.99038735]]


_input = [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]]

xSize = 5
ySize = 5

sum = 0
for subx in [-1,0,1]:
            for suby in [-1,0,1]:

                subbx = subx+x
                if subbx != -1: subbx = subbx%xSize

                subby = suby+y
                if subby != -1: subby = subby%ySize

                print(subbx)

                sum += _input[subbx,subby] * _filter[subx+1,suby+1]
                #sum += _input[(subx+x)%(xSize),(suby+y)%(ySize)] * _filter[subx+1,suby+1]

print(sum)