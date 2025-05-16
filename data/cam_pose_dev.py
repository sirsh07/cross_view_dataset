import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

pan_deg = 20 # pos for right, neg for left
json_path = ".\\google_earth\\ID0058\\ID0058.json"  # Update to your actual path

# === Utility: Geodetic <-> ECEF ===
def geodetic_to_ecef(lat, lon, h):
    a = 6378137.0
    e2 = 6.69437999014e-3
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    N = a / np.sqrt(1 - e2 * np.sin(lat_r)**2)
    X = (N + h) * np.cos(lat_r) * np.cos(lon_r)
    Y = (N + h) * np.cos(lat_r) * np.sin(lon_r)
    Z = (N * (1 - e2) + h) * np.sin(lat_r)
    return np.array([X, Y, Z])

def ecef_to_geodetic(X, Y, Z):
    a = 6378137.0
    e2 = 6.69437999014e-3
    b = a * np.sqrt(1 - e2)
    ep2 = (a**2 - b**2) / b**2
    p = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Z * a, p * b)
    lat = np.arctan2(Z + ep2 * b * np.sin(theta)**3,
                     p - e2 * a * np.cos(theta)**3)
    lon = np.arctan2(Y, X)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    h = p / np.cos(lat) - N
    return np.rad2deg(lat), np.rad2deg(lon), h

# === Load Camera Data ===
with open(json_path, 'r') as f:
    data = json.load(f)

frames = data['cameraFrames']
# Convert each frame's geodetic to ECEF for camera positions
ecefs = np.array([geodetic_to_ecef(cf['coordinate']['latitude'],
                                   cf['coordinate']['longitude'],
                                   cf['coordinate']['altitude'])
                  for cf in frames])

# === Compute Geodetic Center ===
lats = np.array([cf['coordinate']['latitude'] for cf in frames])
lons = np.array([cf['coordinate']['longitude'] for cf in frames])
alts = np.array([cf['coordinate']['altitude'] for cf in frames])
center_lat = lats.mean()
center_lon = lons.mean()
center_alt = alts.mean()
center_ecef = geodetic_to_ecef(center_lat, center_lon, center_alt)

# === Setup ENU frame at center ===
lat0 = np.deg2rad(center_lat)
lon0 = np.deg2rad(center_lon)
R_ecef2enu = np.array([
    [-np.sin(lon0),             np.cos(lon0),              0],
    [-np.sin(lat0)*np.cos(lon0), -np.sin(lat0)*np.sin(lon0), np.cos(lat0)],
    [ np.cos(lat0)*np.cos(lon0),  np.cos(lat0)*np.sin(lon0), np.sin(lat0)]
])

angle = np.deg2rad(pan_deg)
rot_enu = R.from_euler('z', -pan_deg, degrees=True).as_matrix()

# === Compute all deviated look-at ECEF points ===
lookat_ecefs = np.zeros_like(ecefs)
for i, pos in enumerate(ecefs):
    v_ecef = center_ecef - pos
    v_enu = R_ecef2enu.dot(v_ecef)
    horiz = v_enu.copy(); horiz[2] = 0
    dist_h = np.linalg.norm(horiz)
    dir_h = horiz / dist_h
    dir_rot = rot_enu.dot(dir_h)
    v_enu_rot = np.array([dir_rot[0]*dist_h, dir_rot[1]*dist_h, v_enu[2]])
    v_ecef_rot = R_ecef2enu.T.dot(v_enu_rot)
    lookat_ecefs[i] = pos + v_ecef_rot

# === Sample Specific Frame Indices ===
sample_frames = [0,15,30,45,60,90,120,150,180,225,270,315,360]
max_idx = len(lookat_ecefs) - 1
sample_idx = [min(f, max_idx) for f in sample_frames]

print("Corrected Deviated LookAt (lat, lon, alt):")
for f in sample_idx:
    # vector from camera to center in ECEF
    v_ecef = center_ecef - ecefs[f]
    # to ENU
    v_enu = R_ecef2enu.dot(v_ecef)
    # normalize and rotate in ENU horizontal plane
    horiz = v_enu.copy()
    horiz[2] = 0
    dist_h = np.linalg.norm(horiz)
    dir_h = horiz / dist_h
    dir_rot = rot_enu.dot(dir_h)
    # reconstruct rotated ENU vector, keep original vertical component
    v_enu_rot = np.array([dir_rot[0]*dist_h, dir_rot[1]*dist_h, v_enu[2]])
    # back to ECEF
    v_ecef_rot = R_ecef2enu.T.dot(v_enu_rot)
    lookat_ecef = ecefs[f] + v_ecef_rot
    # to geodetic
    lat, lon, alt = ecef_to_geodetic(*lookat_ecef)
    print(f" Frame {f:3d}: lat=,{lat:.6f}, lon=,{lon:.6f}, alt={alt:.2f} m")

# === Plot 3D Trajectory & Deviated LookAt ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Camera positions
ax.scatter(ecefs[:,0], ecefs[:,1], ecefs[:,2],
           c='blue', marker='x', label='Camera Positions')
# Center
ax.scatter(*center_ecef, c='red', s=100, marker='o', label='Center')
# All deviated lookat
ax.scatter(lookat_ecefs[:,0], lookat_ecefs[:,1], lookat_ecefs[:,2],
           c='green', s=20, marker='.', label='All Deviated LookAt')
# Sampled points
ax.scatter(lookat_ecefs[sample_idx,0],
           lookat_ecefs[sample_idx,1],
           lookat_ecefs[sample_idx,2],
           c='red', s=80, marker='o', label='Sampled Frames')

# Draw arrows from camera to sampled look-at
for idx in sample_idx:
    p = ecefs[idx]
    l = lookat_ecefs[idx]
    ax.plot([p[0], l[0]], [p[1], l[1]], [p[2], l[2]],
            color='purple', linewidth=1)

ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
ax.set_title('Corrected 20° Right Deviated LookAt Points')
ax.legend()
ax.set_box_aspect([1,1,1])
plt.show()
