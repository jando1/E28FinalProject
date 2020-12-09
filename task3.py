import sys, os
import numpy as np
import project4
from ursim import RoboSimApp, ctrl, transform2d
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import pf_demo_for_yhja as pf

######################################################################

# Adds the modified new pure pursuit with the cubic

class SlalomController(ctrl.Controller):

    def __init__(self, commands):

        self.commands = commands

    def initialize(self, time, odom_pose):

        self.command_index = 0
        self.gate = None

        #Initialize particle filter
        numParticles = 50
        self.particles = []
        for i in range(numParticles):
            self.particles.append(transform2d.Transform2D(0,0,0))
        self.old_odom = None

        print('will do "{}" action until {}-{} gate is found'.format(
            *self.commands[0]))

    def do_action(self, action, tape_detections):
        # TODO: implement left/right/tape actions

        if action == "left":
            cmd_vel = ctrl.ControllerOutput(.5, np.pi/4)
        
        elif action =="right": 
            cmd_vel = ctrl.ControllerOutput(.5, -np.pi/4)

        else: #tape 
            detections = tape_detections
            distances = np.zeros(len(detections))
            for i in range(len(distances)):
                blob = detections[i]

                pos = blob.xyz_mean    
                distances[i]= np.sqrt(pos[0]**2 + pos[1]**2)
            
            idx = np.argmin(distances) #Find the index of the minimum distance

            pos = detections[idx].xyz_mean 

            kp = 5
            theta_dot = kp * np.arctan2(pos[1],pos[0])

            cmd_vel = cmd_vel = ctrl.ControllerOutput(0.75, theta_dot) #this was modified with higher tape gain from project 4, large time improvement
            
        return cmd_vel

    def find_cubic(self, T_world_from_robot, T_world_from_gate):
        
        T_robot_from_gate = T_world_from_robot.inverse() * T_world_from_gate
        
        robot_in_gate_frame = T_robot_from_gate.inverse().position

        (p,q) = robot_in_gate_frame
        tan_theta = q/p

        #print(tan_theta)

        A = np.array([[p**3, p ** 2],
                      [3 * p ** 2, 2 * p]])
        b = np.array([[q], [tan_theta]])
        
        coeff =linalg.solve(A,b)

        a = coeff[0]
        b = coeff[1]

        x = np.linspace(p, 0, 1000)
        y = a * x ** 3 + b * x ** 2

        """plt.figure()
        plt.plot(x,y)
        
        plt.axis('equal')
        plt.show(block = False)
        input("Press enter to continue")
        plt.close()"""

        return (a,b)

        
    
#Particle Filter

    def particle_filter(self, T_world_from_robot, T_world_from_gate):
        sigma_motion_xy_init = 0.01
        sigma_motion_theta_init = 0.01

        sigma_motion_xy = 0.02
        sigma_motion_theta = 0.01

        sigma_meas_xy = 0.1
        sigma_meas_theta = 0.04


        if self.old_odom is not None:

            T_prev_from_cur = self.old_odom.inverse() * T_world_from_robot

        else:
            T_prev_from_cur = transform2d.Transform2D(0,0,0)



        self.particles = pf.motion_update(self.particles, T_prev_from_cur, sigma_motion_xy, sigma_motion_theta)

        if T_world_from_gate is not None:
            #Measurement Step
            
            T_robot_from_gate = T_world_from_robot.inverse() * T_world_from_gate
    
            self.particles, weights = pf.measurement_update(self.particles, 
                       T_robot_from_gate, T_world_from_gate, 
                       sigma_meas_xy, sigma_meas_theta)
        
        x = []
        y = []
        for idx, particle in enumerate(self.particles):
            x.append(particle.position[0])
            y.append(particle.position[1])

        return self.average_particle(self.particles)

    def average_particle(self, particles):
        x = []
        y = []
        theta = []
        for idx, particle in enumerate(particles):
            x.append(particle.position[0])
            y.append(particle.position[1])
            theta.append(particle.angle)

        return transform2d.Transform2D(np.mean(x), np.mean(y), np.mean(theta))
        
    def pure_pursuit(self, T_world_from_robot, T_world_from_gate):
        is_finished = False
        
        T_world_from_robot_pursuit = self.particle_filter(T_world_from_robot, T_world_from_gate)
        # Establish the position of the robot in the gate coordinate frame. 
        T_robot_from_gate = T_world_from_robot_pursuit.inverse() * T_world_from_gate
        robot_in_gate_frame = T_robot_from_gate.inverse().position

        #print("Robot position in gate frame: ", robot_in_gate_frame)
        alpha = 0.7

        if robot_in_gate_frame[0] > 0:
            is_finished = True
        
        x = robot_in_gate_frame[0] + alpha
        if x > 0:
            y = 0
        else:
            (a,b) = self.find_cubic(T_world_from_robot_pursuit, T_world_from_gate)
            y = a * x ** 3 + b * x ** 2
        
        target_in_gate_frame = np.array([x,y])
        
        target_in_robot_frame=T_robot_from_gate.transform_fwd(target_in_gate_frame)


        k = 2.5
        omega = k * np.arctan2(target_in_robot_frame[1], target_in_robot_frame[0])
        cmd_vel = ctrl.ControllerOutput(0.75, omega)
        
        return cmd_vel, is_finished

    def match_gate(self, T_world_from_robot, gates, lcolor, rcolor, current_gate):

        closest_matching_gate = None
        closest_matching_gate_dist = np.inf
        

        # TODO: choose the gate closest to the robot with the given
        # left & right color

        for gate in gates:
            left_color, right_color, T = gate
            if left_color == lcolor and right_color == rcolor:
                T_robot_from_gate = T_world_from_robot.inverse() * T
                robot_in_gate_frame = T_robot_from_gate.inverse().position
                curr_dist= np.linalg.norm(robot_in_gate_frame)

                if curr_dist < closest_matching_gate_dist: 
                    closest_matching_gate = gate
                    closest_matching_gate_dist = curr_dist
                
        
        # TODO: if there is a current gate and a closest matching gate
        # they are further apart than a reasonable threshold, discard
        # the closest matching gate
        if current_gate is not None and closest_matching_gate is not None: 
            T_world_from_current=current_gate[2]
            position_current = T_world_from_current.position

            T_world_from_closest_matching = closest_matching_gate[2]
            position_closest = T_world_from_closest_matching.position

            threshold = 1
            if np.linalg.norm(position_current-position_closest) >threshold: 
                closest_matching_gate = current_gate

        if current_gate is not None and closest_matching_gate is None:
            closest_matching_gate = current_gate

        return closest_matching_gate
        
    def update(self, time, dt, robot_state, camera_data):

        T_world_from_robot = robot_state.odom_pose
        oldpose = robot_state.odom_pose
        gates = project4.find_gates(T_world_from_robot, camera_data.detections)
        #print(gates)

        if self.command_index >= len(self.commands):

            # all finished!
            return None

        else:

            action, lcolor, rcolor = self.commands[self.command_index]

            detected_gate = self.match_gate(T_world_from_robot, gates, lcolor, rcolor, self.gate)

            if detected_gate is not None:
                if self.gate is None:
                    print('found {}-{} gate'.format(lcolor, rcolor))
                    
                    
                self.gate = detected_gate
            #Do particle
            if self.gate is None:
                self.particle_filter(T_world_from_robot, None)
                cmd_vel = self.do_action(action, camera_data.detections['blue_tape'])

            else:

                T_world_from_gate = self.gate[2]

                cmd_vel, is_finished = self.pure_pursuit(T_world_from_robot,
                                                         T_world_from_gate)

                if is_finished:
                    
                    print('passed through {}-{} gate'.format(self.gate[0], self.gate[1]))
                    self.gate = None
                    self.command_index += 1
                    
                    if self.command_index == len(self.commands): 
                        print('finished the course!')
                    else:
                        print('will do "{}" action until {}-{} gate is found'.format(
                            *self.commands[self.command_index]))

            self.old_odom = oldpose
            return cmd_vel
        
######################################################################

def main():

    if len(sys.argv) != 3 or not os.path.exists(sys.argv[1]) or not os.path.exists(sys.argv[2]):
        print('usage: {} MAP.svg COMMANDS.txt'.format(os.path.basename(sys.argv[0])))
        sys.exit(1)

    commands = project4.load_commands(sys.argv[2])
        
    app = RoboSimApp(SlalomController(commands))

    app.sim.load_svg(sys.argv[1])

    app.sim.add_observer(project4.Referee(commands))

    app.run()


if __name__ == '__main__':
    main()

    
