import sys, os
import numpy as np
import project4
from ursim import RoboSimApp, ctrl, transform2d
import numpy.linalg as linalg
import matplotlib.pyplot as plt

######################################################################

# Adds the modified new pure pursuit with the cubic

class SlalomController(ctrl.Controller):

    def __init__(self, commands):

        self.commands = commands

    def initialize(self, time, odom_pose):

        self.command_index = 0
        self.gate = None

        #Initialize particle filter
        numParticles = 20
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

    # Given an array of particles x and an action u, return a new array that
    # is the result of the motion step of the particle filter.
    def motion_update(self, x, u, sigma_x):
        # TODO: add the action to the state array to get the un-corrupted
        # new states
        x_new = []
        for idx, particle in enumerate(x):
            new_particle = u.transform_fwd(particle)
            """ u = np.expand_dims(u,axis=1)
            u = np.multiply(np.ones_like(x), u)
            x_new= x+ u """
            print("PARTICLE UPDATED TO NEW STATE")
            noise =  np.random.normal(0,sigma_x)
            #print(noise)
            new_particle.position[0] += np.random.normal(0,sigma_x)
            new_particle.position[1] += np.random.normal(0,sigma_x)
            new_particle.angle += np.random.normal(0,sigma_x)
            x_new.append(new_particle)

        return x_new

    # Given an array of particles x and a measurement z, return a new array
    # that is the result of the measurement step of the particle filter.
    def measurement_update(self, x, z, sigma_z, T_world_from_robot, T_world_from_gate):
        if T_world_from_gate is None:
            weights = np.ones(np.shape(x))
            print("weights when no measurement: ", weights)
            weights /= np.sum(weights)


            x=np.random.choice(x, p=weights, size=len(weights))
            #print(x)
            return x
        # TODO: compute weights for each particle by evaluating the Gaussian
        # probability density function associated with the sensor model.
        #z = np.expand_dims(z, axis = 1)
        #print(x)
        #gate_in_world_frame = T_world_from_gate.inverse().position
        #gate_in_robot_frame = T_world_from_robot.transform_inv(gate_in_world_frame)

        #T_robot_from_gate = T_world_from_robot.inverse() * T_world_from_gate
        #gate_in_robot_frame = np.array([T_robot_from_gate.x, T_robot_from_gate.y, T_robot_from_gate.theta])
        
        print("robot in gate frame: ", z)
        weights = []
        #print("empty weights:", weights)
        for idx, particle in enumerate(x):

            T_particle_from_gate = T_world_from_gate.transform_inv(particle)
            particle_in_gate_frame = np.array([T_particle_from_gate.position[0], T_particle_from_gate.position[1], T_particle_from_gate.angle])
            print('particle in gate frame: ', particle_in_gate_frame)
            cur_weights = 1/(sigma_z * np.sqrt( 2 * np.pi)) * np.exp( - (particle_in_gate_frame - z) ** 2 /(2 * sigma_z ** 2))
            weights.append(np.prod(cur_weights))
        #cur_weights=np.multiply(weights[0,:],weights[1,:])  
        weights = np.array(weights) 
       # print("weights unnormalized:", weights) 
        weights /= np.sum(weights)
        #print("weights normalized:", weights)

        x=np.random.choice(x, p=weights, size=len(weights))
        #print(x)
        return x

    def particle_filter(self, T_world_from_robot, T_world_from_gate):
        sigma_x = .1
        sigma_y = 0.05
        new_position = T_world_from_robot

        if self.old_odom is not None:
            """old_position = self.old_odom.position

            print("old position: ",old_position)
            print("new position: ", new_position)

            action = new_position - old_position"""

            T_cur_from_previous = T_world_from_robot.inverse() * self.old_odom
            action=T_cur_from_previous

        else:
            action = transform2d.Transform2D(0,0,0)


        print('action: ',action)
        
        #print("robot position in world frame: ", T_world_from_robot)

        #Motion Step
        #print("x from odometry: ", T_world_from_robot.position)
        new_particles = self.motion_update(self.particles, action, sigma_x)
        #print("particles after motion step: ", self.particles)
        #print("T world from gate: ",T_world_from_gate)
        if T_world_from_gate is not None:
            #Measurement Step
            #
            
            T_robot_from_gate = T_world_from_robot.inverse() * T_world_from_gate
            T_robot_in_gate_frame = T_robot_from_gate.inverse()
            gate_in_robot_frame = np.array([T_robot_in_gate_frame.position[0], T_robot_in_gate_frame.position[1], T_robot_in_gate_frame.angle])

            measurement = gate_in_robot_frame
            #print("measurement", measurement)
            #print("mean position", np.mean(self.particles, axis = 1))
            self.particles = self.measurement_update(new_particles, measurement, sigma_y, T_world_from_robot, T_world_from_gate)
            #print("particles after measurement step:", self.particles)
        else:
            print("NO MEASUREMENT")
            self.particles = self.measurement_update(new_particles, None, sigma_y, T_world_from_robot, T_world_from_gate)
            print(self.particles)
        
        x = []
        y = []
        for idx, particle in enumerate(self.particles):
            x.append(particle.position[0])
            y.append(particle.position[1])

        print("robot position from odometery: ", T_world_from_robot.position)
        print("robot position from particle filter: ", np.array([np.mean(x), np.mean(y)]))

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
        #print(robot_in_gate_frame[1])
        #print("Robot angle cotangent in gate frame:", robot_in_gate_frame[1]/robot_in_gate_frame[0])

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
        #print("HERE!!!!!!!!")
        #self.robot_state = robot_state
        print("robot state at beginning: ", oldpose.position)
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
            print("Defined old odom as", oldpose.position)
            print("******************")
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

    
