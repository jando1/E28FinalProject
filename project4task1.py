import sys, datetime
import numpy as np
import project4
from ursim import RoboSimApp, ctrl
from ursim.transform2d import Transform2D

######################################################################

class PracticeController(ctrl.Controller):

    def __init__(self):
        super().__init__()

    def initialize(self, time, odom_pose):
        self.gate = None

    def pure_pursuit(self, T_world_from_robot, T_world_from_gate):
        
        is_finished = False
        # TODO: implement the pure pursuit algorithm we discussed in class

        # Establish the position of the robot in the gate coordinate frame. 
        T_robot_from_gate = T_world_from_robot.inverse() * T_world_from_gate
        robot_in_gate_frame = T_robot_from_gate.inverse().position

        print("Robot position in gate frame: ", robot_in_gate_frame)

        if robot_in_gate_frame[0] > 0:
            is_finished = True

        alpha = 0.7
        target_in_gate_frame = np.array([robot_in_gate_frame[0] + alpha, 0])
        
        target_in_robot_frame=T_robot_from_gate.transform_fwd(target_in_gate_frame)


        k = 2.5
        omega = k * np.arctan2(target_in_robot_frame[1], target_in_robot_frame[0])
        cmd_vel = ctrl.ControllerOutput(0.75, omega)
        

        return cmd_vel, is_finished
        
    def update(self, time, dt, robot_state, camera_data):

        T_world_from_robot = robot_state.odom_pose
        
        gates = project4.find_gates(T_world_from_robot, camera_data.detections)

        if len(gates):
            self.gate = gates[0]
        
        if self.gate is None:
            
            return None

        else:

            lcolor, rcolor, T_world_from_gate = self.gate

            cmd_vel, is_finished = self.pure_pursuit(T_world_from_robot,
                                                     T_world_from_gate)

            if is_finished:
                print('all done!')
                self.gate = None

            return cmd_vel
        
######################################################################

def main():
    
    if len(sys.argv) == 1:
        seed = int(datetime.datetime.now().timestamp() * 1000) % 100000
    else:
        seed = int(sys.argv[1])

    print('using seed', seed)
    np.random.seed(seed)
    
    app = RoboSimApp(PracticeController())

    project4.setup_practice_task(app.sim, randomize=True)

    app.sim.add_observer(project4.Referee())

    app.run()


if __name__ == '__main__':
    main()

            

