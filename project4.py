import sys, os
import numpy as np

from ursim import ctrl, RoboSimApp
from ursim.transform2d import Transform2D
from ursim.core import SimObserver, Pylon
from ursim.core import ROBOT_BASE_RADIUS, PYLON_RADIUS

MIN_GATE_DISTANCE = 0.55 # m
MAX_GATE_DISTANCE = 0.75 # m

DEFAULT_ROOM_DIMS = np.array([8., 6.])
DEFAULT_LPOS = np.array([6., 3.33])
DEFAULT_RPOS = np.array([6., 2.67])

ROBOT_POS_MIN = np.array([2.0, 2.0])
ROBOT_POS_MAX = np.array([3.0, 4.0])

ROBOT_ANGLE_MAX_PERTURB = 15*np.pi/180

VALID_COMMANDS = ['left', 'right', 'tape']
VALID_COLORS = ['orange', 'green']

MAX_TIME_PER_COMMAND = 10 # seconds

######################################################################

def load_commands(txtfile):
    """Parse the text file with the given filename and return a list 
    of (action, left_color, right_color) tuples."""

    print(txtfile)
    commands = []

    with open(txtfile, 'r') as istr:
        for line in istr:
            line = line.strip()
            if not line:
                continue
            tokens = line.split(' ')
            assert len(tokens) == 3
            assert tokens[0] in VALID_COMMANDS
            assert tokens[1] in VALID_COLORS
            assert tokens[2] in VALID_COLORS
            commands.append(tuple(tokens))

    return commands

######################################################################

def get_gate_xform(lgoal_pos, rgoal_pos):
    """Given the positions of the gate pylons in some reference coordinate
    frame, construct the rigid transformation T_reference_from_gate,
    where the gate coordinate frame is defined as having its center
    point at the midpoint between the gate pylons, the x-axis pointing
    perpendicular to the line between the pylons, and the y-axis
    pointing from the right pylon to the left one.

    Essentially identical to get_goal_xform in Project 2."""

    pmid = 0.5 * (lgoal_pos + rgoal_pos)

    diff = lgoal_pos - rgoal_pos

    d_sin_theta = -diff[0]
    d_cos_theta =  diff[1]

    theta = np.arctan2(d_sin_theta, d_cos_theta)

    return Transform2D(pmid, theta)

######################################################################

def find_gates(T_world_from_robot, detections):

    """Given the current robot pose and a list of camera detections,
    return a list of (left_color, right_color, T_world_from_gate)
    tuples."""

    if detections is None:
        return []

    ##################################################
    # build a list of (color, position in robot frame) tuples 

    all_pylons = []

    for color in VALID_COLORS:

        cdetections = detections[color + '_pylon']

        for pylon in cdetections:
            pylon_in_robot_frame = pylon.xyz_mean[:2]
            all_pylons.append( (color, pylon_in_robot_frame) )

    ##################################################
    # sort list by increasing y coordinate in robot frame

    all_pylons.sort(key=lambda color_pos: color_pos[1][1])

    ##################################################
    # iterate over all pairs of pylons

    gates = []
    
    for i, lpylon in enumerate(all_pylons):

        lcolor, lpos = lpylon
        
        for rpylon in all_pylons[:i]:

            rcolor, rpos = rpylon

            assert lpos[1] >= rpos[1]

            lr_distance = np.linalg.norm(lpos - rpos)

            if (lr_distance >= MIN_GATE_DISTANCE and
                lr_distance <= MAX_GATE_DISTANCE):

                T_robot_from_gate = get_gate_xform(lpos, rpos)

                T_world_from_gate = T_world_from_robot * T_robot_from_gate

                dist_from_me = np.linalg.norm(T_robot_from_gate.position)

                gates.append( (dist_from_me, lcolor, rcolor, T_world_from_gate) )

    ##################################################
    # sort gates by distance and get rid of distance

    gates = [ gate[1:] for gate in sorted(gates) ]

    return gates
    
######################################################################

def rand_vec(vmin, vmax):
    """Utility function used by setup_practice_task to sample within a box."""
    return vmin + np.random.random(size=vmin.shape)*(vmax-vmin)

def angle_from_vec(v):
    """Construct the angle between the vector v and the x-axis."""
    return np.arctan2(v[1], v[0])

######################################################################

class GateMonitor:

    """Helper class to judge when the robot has moved or passed through a
    gate. The Referee object (see below) constructs one GateMonitor
    per detected gate in the slalom course.
    """

    def __init__(self, lpylon, rpylon):

        self.pylons = [lpylon, rpylon]

        lpos = lpylon.orig_position
        rpos = rpylon.orig_position

        distance = np.linalg.norm(lpos - rpos)

        self.gate_radius = 0.5*distance

        assert (distance >= MIN_GATE_DISTANCE and
                distance <= MAX_GATE_DISTANCE)

        self.lcolor = lpylon.color_name
        self.rcolor = rpylon.color_name

        self.T_room_from_gate = get_gate_xform(lpos, rpos)
        self.T_gate_from_room = self.T_room_from_gate.inverse()

        self.reset()

    def reset(self):
        self.is_touched = False

    def update(self, prev_robot_room_pos, cur_robot_room_pos):

        if self.is_touched:
            return None
            
        for pylon in self.pylons:
            move = pylon.orig_position - pylon.body.position
            move_dist = np.linalg.norm(move)
            if move_dist > 1e-4:
                self.is_touched = True
                return None

        prev_robot_gate_pos = self.T_gate_from_room * prev_robot_room_pos
        cur_robot_gate_pos = self.T_gate_from_room * cur_robot_room_pos

        p = PYLON_RADIUS
        r = ROBOT_BASE_RADIUS
        g = self.gate_radius

        abs_y = abs(cur_robot_gate_pos[1])
        sign_prev_x = np.sign(prev_robot_gate_pos[0])
        sign_cur_x = np.sign(cur_robot_gate_pos[0])

        if abs_y <= g - p - r and sign_prev_x != sign_cur_x:
            if sign_prev_x <= 0:
                return 1
            else:
                return -1
        else:
            return 0
                
############################################################            

class Referee(SimObserver):

    """Class to determine the success/failure of slalom tasks. For
    practice, the robot must pass through the gate within 15 seconds;
    for a slalom course, the robot is allocated 15 seconds per command
    in the course description. For both types of trial, the robot is
    disqualified if it moves a gate pylon or the bumper is activated.
    For the slalom course, the robot is also disqualified if it
    traverses gates in the incorrect order.""" 

    def __init__(self, commands=None):
        """Initializes the Referee - pass in commands=None to do practice mode."""
        super().__init__()
        self.commands = commands

    def register(self, sim):
        """Registers logging variables."""
        
        variable_names = [
            'slalom.success',
            'slalom.robot_disqualified',
            'slalom.disqualify_reason',
            'slalom.last_gate',
        ]

        self.log_vars = np.zeros(len(variable_names), dtype=np.float32)

        sim.datalog.add_variables(variable_names, self.log_vars)

        sim.datalog.register_enum(
            'slalom.disqualify_reason',
            ['N/A', 'hit gate', 'wrong order', 'bumper activated', 'timeout'])

        gate_strs = ['N/A']
        self.gate_str_lookup = dict()

        for lcolor in VALID_COLORS:
            for rcolor in VALID_COLORS:
                gstr = '{}, {}'.format(lcolor, rcolor)
                self.gate_str_lookup[(lcolor, rcolor)] = len(gate_strs)
                gate_strs.append(gstr)

        sim.datalog.register_enum(
            'slalom.last_gate',
            gate_strs)

    def initialize(self, sim):

        """Called when sim initialized/reset to create gate monitors, etc."""

        pylons = []

        for obj in sim.objects:
            if isinstance(obj, Pylon):
                pylons.append(obj)

        self.gate_monitors = []

        for i, pylon_a in enumerate(pylons):
            pos_a = pylon_a.orig_position
            for pylon_b in pylons[:i]:
                pos_b = pylon_b.orig_position
                diff = pos_b - pos_a
                dist = np.linalg.norm(diff)
                if (dist >= MIN_GATE_DISTANCE and
                    dist <= MAX_GATE_DISTANCE):
                    self.gate_monitors.append(GateMonitor(pylon_a, pylon_b))

        self.prev_robot_room_pos = sim.robot.body.position

        self.command_index = 0
        self.log_vars[:] = 0

        self.timeout = MAX_TIME_PER_COMMAND * (1 if self.commands is None else len(self.commands))
        
        print('*** REF SAYS: putting {} seconds on the clock ***'.format(
            self.timeout))

        if self.commands is not None:
            print('*** REF SAYS: first gate to pass through is {}-{} ***'.format(
                *self.commands[0][1:]))
        
    def post_update(self, sim):

        """Called after every sim update to determine success/failure."""

        cur_robot_room_pos = sim.robot.body.position

        if self.log_vars[0] or self.log_vars[1]:
            return

        for gm in self.gate_monitors:
            
            status = gm.update(self.prev_robot_room_pos, cur_robot_room_pos)
            
            if status is None:

                print('*** REF SAYS: {}-{} gate was moved! :( ***'.format(
                    gm.lcolor, gm.rcolor))
                
                self.log_vars[1] = 1
                self.log_vars[2] = 1
                
            elif status != 0:
                
                # passed thru a gate
                if status == 1:
                    gate_colors = (gm.lcolor, gm.rcolor)
                else:
                    assert status == -1
                    gate_colors = (gm.rcolor, gm.lcolor)

                print('*** REF SAYS: robot passed through {}-{} gate! ***'.format(
                    *gate_colors))
                    
                self.log_vars[3] = self.gate_str_lookup[gate_colors]
                    
                if self.commands is None:

                    self.log_vars[0] = 1

                else:

                    if self.command_index <= len(self.commands):
                        cur_command = self.commands[self.command_index]
                        cur_gate = cur_command[1:]
                        if gate_colors != cur_gate:
                            print('*** REF SAYS: wrong gate, '
                                  'was expecting {}-{} :( ***'.format(*cur_gate))
                            self.log_vars[1] = 1
                            self.log_vars[2] = 2
                        else:
                            self.command_index += 1
                            if self.command_index == len(self.commands):
                                self.log_vars[0] = 1
                            else:
                                print('*** REF SAYS: next gate '
                                      'to pass through is {}-{} ***'.format(
                                          *self.commands[self.command_index][1:]))

        if self.log_vars[0]:

            total_time = sim.sim_time.total_seconds()
            pct_time = 100 * total_time / self.timeout

            print('*** REF SAYS: succesfully completed '
                  'after {:.2f} seconds - {:.1f}% of allotted time :) ***'.format(
                      total_time, pct_time))

        elif not self.log_vars[1]:
            
            if np.any(sim.robot.bump):
                self.log_vars[1] = 1
                self.log_vars[2] = 3
                print('*** REF SAYS: bumper activated :( ***')
            elif sim.sim_time.total_seconds() > self.timeout:
                self.log_vars[1] = 1
                self.log_vars[2] = 4
                print('*** REF SAYS: timeout :( ***')

        self.prev_robot_room_pos = np.array(cur_robot_room_pos).copy()

######################################################################

def setup_practice_task(sim, randomize):

    """Set up a small room with a single gate and possibly randomize the 
    robot's position and orientation relative to the gate."""

    # create room
    sim.set_dims(*DEFAULT_ROOM_DIMS)

    # create goalposts
    sim.add_pylon(DEFAULT_RPOS, VALID_COLORS[np.random.randint(2)])
    sim.add_pylon(DEFAULT_LPOS, VALID_COLORS[np.random.randint(2)])

    if randomize:
        robot_pos = rand_vec(ROBOT_POS_MIN, ROBOT_POS_MAX)
    else:
        robot_pos = 0.5*(ROBOT_POS_MIN + ROBOT_POS_MAX)

        
    center_pos = 0.5*(DEFAULT_LPOS + DEFAULT_RPOS)
    
    heading = angle_from_vec(center_pos - robot_pos)

    if randomize:
        heading += (np.random.random()*2-1)*ROBOT_ANGLE_MAX_PERTURB

    sim.initialize_robot(robot_pos, heading)


######################################################################

class BlastForwardController(ctrl.Controller):

    """Dummy controller that just always drives forward fast."""

    def __init__(self):
        """Initializer just calls base class initializer."""
        super().__init__()

    def update(self, time, dt, robot_state, camera_data):
        """Go fast!!!"""
        return ctrl.ControllerOutput(forward_vel=0.7, angular_vel=0)

######################################################################

def main():
    
    app = RoboSimApp(BlastForwardController())
    setup_practice_task(app.sim, randomize=False)
    app.sim.add_observer(Referee())
    app.run()
    

if __name__ == '__main__':
    main()
    
