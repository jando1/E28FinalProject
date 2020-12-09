import numpy as np
from ursim.transform2d import Transform2D
import matplotlib.pyplot as plt
import scipy.stats

num_particles = 50

sigma_motion_xy_init = 0.01
sigma_motion_theta_init = 0.01

sigma_motion_xy = 0.02
sigma_motion_theta = 0.01

sigma_meas_xy = 0.1
sigma_meas_theta = 0.04


# Tasks:
#
# 1) Implement motion update. Verify that when sigmas are set to zero,
#    the motion of all the particles is exactly the same as the motion
#    of the robot in the plot.
#
# 2) Implement measurement update.
#
# 3) Play with the sigmas to make sure that things are behaving as you
#    expect.
#
# 4) Pull the motion_update and measurement_update functions into your
#    own controller code.

# Given an array of particles x and an action u, return a new array that
# is the result of the motion step of the particle filter.
def motion_update(particles, T_prev_from_cur, sigma_motion_xy, sigma_motion_theta):

    # TODO: for each T_world_from_prev in particles
    #
    #   compute new T_world_from_cur based on particle xform and rel xform passed in
    #   randomly perturb T_world_from_cur based on sigmas passed in
    #   append to new particles array
    new_particles = []
    for particle in particles:
        T_world_from_cur = particle *T_prev_from_cur
    
        T_world_from_cur.position[0] += np.random.normal(0,sigma_motion_xy)
        T_world_from_cur.position[1] += np.random.normal(0,sigma_motion_xy)
        T_world_from_cur.angle += np.random.normal(0,sigma_motion_theta)

        new_particles.append(T_world_from_cur)

    return new_particles

# return particles and weights
def measurement_update(particles, 
                       T_robot_from_gate_meas, T_world_from_gate_true, 
                       sigma_meas_xy, sigma_meas_theta):

    # TODO: compute weights. for each T_world_from_robot in particles,
    #   
    #  - compute T_robot_from_gate_expected based on particle xform
    #    and T_world_from_gate_true
    #
    #  - use sigmas to compute p(x), p(y), and p(theta) based on comparing
    #    measured and expected T_robot_from_gate transforms and sigmas
    #    passed in
    # 
    #  - weight should be product of p(x) * p(y) * p(theta)
    #
    # Then resample as you would for any particle fitlers
    weights=[]
    for particle in particles:
        T_robot_from_gate_expected = particle.inverse() * T_world_from_gate_true 
        weight_x = scipy.stats.norm.pdf(T_robot_from_gate_expected.position[0], T_robot_from_gate_meas.position[0], sigma_meas_xy) 
        weight_y = scipy.stats.norm.pdf(T_robot_from_gate_expected.position[1], T_robot_from_gate_meas.position[1], sigma_meas_xy) 
        weight_theta = scipy.stats.norm.pdf(T_robot_from_gate_expected.angle, T_robot_from_gate_meas.angle, sigma_meas_theta) 
        weight = weight_x * weight_y * weight_theta
        weights.append(weight)

    weights = np.array(weights)    
    weights /= np.sum(weights)
    particles = np.random.choice(particles, p=weights, size=len(weights))
    
    return particles, weights

def draw_xform(xform, **kw):
    x0, y0 = xform.position
    x1, y1 = xform * (0.2, 0)
    plt.quiver([x0], [y0], [x1-x0], [y1-y0], **kw)

def get_cmd_velocities(t):

    forward = 1.0*np.sin(16*t)
    angular = 1.0*np.sin(4*t)

    return forward, angular

def main():
    
    t = 0.0
    dt = 0.04

    T_world_from_gate = Transform2D((3, 1), 0.5)
    
    T_world_from_robot_prev = Transform2D((0, 0), 0)

    particles = []
    for i in range(num_particles):
        x = T_world_from_robot_prev.copy()
        x.position[0] += np.random.normal(scale=sigma_motion_xy_init)
        x.position[1] += np.random.normal(scale=sigma_motion_xy_init)
        x.angle += np.random.normal(scale=sigma_motion_xy_init)
        particles.append(x)

    is_motion_step = True

    while True:

        plt.clf()

        if is_motion_step:

            t += dt
            fwd_vel, ang_vel = get_cmd_velocities(t)

            xnew = T_world_from_robot_prev.position[0] + fwd_vel * np.cos(T_world_from_robot_prev.angle) * dt
            ynew = T_world_from_robot_prev.position[1] + fwd_vel * np.sin(T_world_from_robot_prev.angle) * dt
            thetanew = T_world_from_robot_prev.angle + ang_vel * dt

            T_world_from_robot_cur = Transform2D((xnew, ynew), thetanew)
            print(t, fwd_vel, ang_vel, T_world_from_robot_cur)

            T_prev_from_cur = T_world_from_robot_prev.inverse() * T_world_from_robot_cur
            
            # simulate odometry noise
            T_prev_from_cur.position[0] += np.random.normal(scale=sigma_motion_xy)
            T_prev_from_cur.position[1] += np.random.normal(scale=sigma_motion_xy)
            T_prev_from_cur.angle += np.random.normal(scale=sigma_motion_theta)

            particles = motion_update(particles, T_prev_from_cur, sigma_motion_xy, sigma_motion_theta)
            print('did motion update!')

            T_world_from_robot_prev = T_world_from_robot_cur

        else:

            T_robot_from_gate_meas = T_world_from_robot_cur.inverse() * T_world_from_gate

            T_robot_from_gate_meas.position[0] += np.random.normal(scale=sigma_meas_xy)
            T_robot_from_gate_meas.position[1] += np.random.normal(scale=sigma_meas_xy)
            T_robot_from_gate_meas.angle += np.random.normal(scale=sigma_meas_theta)

            T_world_from_gate_meas = T_world_from_robot_cur * T_robot_from_gate_meas

            particles, weights = measurement_update(particles,
                                                    T_robot_from_gate_meas,
                                                    T_world_from_gate,
                                                    sigma_meas_xy,
                                                    sigma_meas_theta)

            print('did measurement update!')

            draw_xform(particles[weights.argmax()], color='m', zorder=3)
            draw_xform(T_world_from_gate_meas, color='g', zorder=2)

            
        

        for x in particles:
            draw_xform(x, color='k', width=0.002, zorder=1)

        draw_xform(T_world_from_gate, color='r', zorder=1)
        draw_xform(T_world_from_robot_cur, color='b', zorder=2)

        
        
        plt.plot([-1, -1, 4, 4], [-1, 3, -1, 3], 'w.')
        plt.axis('equal')
        plt.title('click to quit or hit space to continue')
        plt.xlabel('gate is red, meas is green, true pos is blue, particles are black, highest weight is purple')
        plt.draw()

        if not plt.waitforbuttonpress():
            break

        is_motion_step = not is_motion_step

        
if __name__ == '__main__':
    main()
