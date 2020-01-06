import numpy as np
import argparse

parser = argparse.ArgumentParser(description= "Calculates optimum Jean's Length and \
                                 density for a user-defined TORCH simulation. \
                                 Values to be set are simulation box size and \
                                 maximum refinement level."
                                 )
#parser.print_help()
parser.add_argument("-b", "--box_size", default=None, required=True, type=float,
                    help="Side length of simulation space in parsecs.")
parser.add_argument("-r", "--refinement_max", default=None, required=True, type=float,
                    help="Maximum level of refinement.")

def jeans_length(box_size, ref_max):
    """Calculates minimum Jean's length for a TORCH simulation 
    (4 cells wide at highest level of refinement).
    Assumes 16x16x16 cells within each FLASH block"""
    
    nxb = 8.0
    cell_req = 2.5 # Refinement requirement enforced by sink particle routine (Federrath C. 2010)
    cs = 2.0e4     # Gas soundspeed cm/s
    G = 6.67259e-8 # Gravitational constant cm^3 / g s^2
    
    xb = (box_size) / (2**(ref_max-1))
    xc = xb/nxb
    jlength = xc * cell_req
    jl_in_pc = "Minimum Jean Length = {0:.6E} pc.".format(jlength)
    jl_in_cm = "Minimum Jean Length = {0:.6E} cm.".format(jlength * 3.08567758128E+18)
    
    jdens = (np.pi * cs**2) / (G * (jlength * 3.08567758128E+18)**2)
    jdens_cgs = "Corresponding density for sink creation = {0:.6E} g/cm^3.".format(jdens)
    print "\n For sim box side: {0:.2f} and max refinement lvl: {1:.0f} \n".format(box_size, ref_max)
    print jl_in_cm
    print jdens_cgs

args = parser.parse_args()

bs = args.box_size
ref = args.refinement_max

jeans_length(bs, ref)
