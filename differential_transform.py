import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
import numpy as np
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from pytorch3d.transforms import Transform3d

num_pts = 5000

mat = np.eye(4)
mat[:3, 3] = np.array([0.5, 1, 2])

#print (mat)

src_pts = np.random.randn(num_pts, 3)

dest_pts = []

"""
for pt in src_pts:
	dest_pts.append(np.dot(mat, np.append(pt, [1])))

dest_pts = np.array(dest_pts)[:, :3]"""

noise = np.random.normal(0, 0.06, (num_pts, 3))

for pt in src_pts:
	dest_pts.append(pt + mat[:3, 3])

noisy_src_pts = src_pts + noise

print (src_pts[0])
print (dest_pts[0])

#exit()

src_pts_tensor = torch.Tensor(src_pts).view(1, num_pts, 3)
dest_pts_tensor = torch.Tensor(dest_pts).view(1, num_pts, 3)
noisy_src_pts_tensor = torch.Tensor(noisy_src_pts).view(1, num_pts, 3)

#print (src_pts_tensor)
#print (dest_pts_tensor)

#print (chamfer_distance(src_pts_tensor, dest_pts_tensor))

trans_params = torch.randn(1, 3, requires_grad=True)
rot_params = torch.randn(1, 3)

opt = torch.optim.Adam([trans_params], lr=1e-1)
tol = 6e-3

for _ in range(500000):

	t = Transform3d().translate(trans_params)

	opt.zero_grad()
	_pts = t.transform_points(noisy_src_pts_tensor)

	loss, _ = chamfer_distance(dest_pts_tensor, _pts)

	loss.backward()

	opt.step()

	print (loss)
	if (loss.item() < tol):
		print ("optimizer met tolerance")
		break

print ()
print (trans_params.detach().cpu().numpy())
print (mat[:3, 3])

new_mat = np.eye(4)
new_mat[:3, 3] = trans_params.detach().cpu().numpy()[0]

dest_pts = []

for pt in src_pts:
	dest_pts.append(pt + trans_params.detach().cpu().numpy()[0])

dest_pts = np.array(dest_pts)[:, :3]

print ()
#print (src_pts)
#print (dest_pts)