"""Unmodified legacy curve extracted from the bundled common.py.
Source SHA256: 96baadd7f37ef53ede20e6203005ce38ca199fa6ac13e4458c45daa1402f3d87
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def rejoin(measured,previous_command,future_plan):
 p=np.asarray(future_plan,np.float64).copy();n=len(p)
 if np.array_equal(measured,previous_command):return p
 dt=measured[:3,3]-previous_command[:3,3]
 dr=Rotation.from_matrix(measured[:3,:3]@previous_command[:3,:3].T)
 sl=Slerp([0,1],Rotation.concatenate([dr,Rotation.identity()]))
 for i in range(n):
  a=(i+1)/n;p[i,:3,3]+=(1-a)*dt;p[i,:3,:3]=sl(a).as_matrix()@p[i,:3,:3]
 return p
