# Project Overview: Micromouse Challenge #

MTRN4110 requried groups of 4 to design a UGV to traverse through a randomly generated maze where the finish position is always
located in the centre of the maze. This project was broken into two phases, Phase A and Phase B. There were two assessed
events that required two different bits of program. The first run would be a vision guided run and the second run required
the UGV to build a map as it progressed. 

## === Phase A === ##
Phase A consists of 4 milestones that were asssessed by mid term.

- Locomotion
    This section was required to develop the control algorithm for the driving component of the UGV.
- Hardware Construction
    This section was required to develop a physical chassis for the UGV to mount the sensors and microcontroller
- Sensing
    This section was required to program the sensing components for the UGV
- Communication
    This section was responsible for creating the communication framework to be used between a laptop and the microcontroller
    to communication instructions and maze layouts
    
I was responsible for the Hardware Construction of the UGV and completed most of the work using hand tools and using the
plywood that was given as part of the kit.
    
## === Phase B === ##

Phase B consists of 4 milestones there were assessed before the end of term

- Driving
    Further refinement of the locomotion code to assist in driving through a small maze.
- Planning
    Path planning when supplied with information about the maze layout.
- Exploration
    Exploring and building the maze as well as planning the path to the centre
- Vision
    Detecting the maze and starting position of the UGV and passing the inforation to the Planning software.
    
I was responsible for the vision code, the output of the code can be seen through the JPG files in this repository.
The basic code flow was to:
  Take a snapshot of the maze.
  Use edge detection and image adjustment and confidence logic to determine the existence of a wall.
  Package the data into a string to be sent to the planning program via Bluetooth.
  Trace the movement of the vehicle and plot the path that it took.
  
## === Outcome === ##

Unfortunately, due to a collision with another UGV during a trial run, the LiPo battery shorted and damaged the read/write port
for the Arduino 2560. The UGV completed the vision run, but was unable to upload the exploration code onto the Arduino. The
faulty battery also messed with the accuracy of the motors which as a result skewed the timer that the driving program used
to turn the UGV. However, despite the issues faced with the testing the project was a success as our team developed a
functioning UGV from scratch.

## === Link to Video of Testing Day === ##

https://drive.google.com/drive/u/0/folders/1wjKKBf2HDaTtvQ7lccNTyUSgRuBu_hmZ
  
   
