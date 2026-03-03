import torch
import torch.optim as optim
device = torch.device("cuda")
positions = torch.tensor([list(map(float, line.strip().split())) for line in open("positions.txt")],dtype=torch.float32,device=device)
p1, p2, p3 = positions[0], positions[1], positions[2]
v1 = p2 - p1
v2 = p3 - p1
normal = torch.cross(v1, v2, dim=0)
normal /= torch.norm(normal)
nx, ny, nz = normal
theta_z = -torch.atan2(nx, ny)
cos_z, sin_z = torch.cos(theta_z), torch.sin(theta_z)
rotated_x = cos_z * nx + sin_z * ny
rotated_y = -sin_z * nx + cos_z * ny
rotated_z = nz
theta_x = torch.atan2(rotated_z, rotated_y)
theta_x_deg = torch.rad2deg(theta_x)
theta_z_deg = torch.rad2deg(theta_z)
print(f"Rotate around X-axis by {theta_x_deg.item():.2f} degrees")
print(f"Rotate around Z-axis by {theta_z_deg.item():.2f} degrees")
target_normal = torch.tensor([0.0, 1.0, 0.0],device=device)
normal=normal.to(device)
theta_z = torch.nn.Parameter(torch.tensor(0.0, device=device, requires_grad=True))
theta_x = torch.nn.Parameter(torch.tensor(0.0, device=device, requires_grad=True))
######################################################
optimizer = optim.Adam([theta_z, theta_x], lr=0.001)
#######################################################
def loss_fn():
    Rz = torch.stack([
        torch.cos(theta_z), -torch.sin(theta_z), torch.tensor(0.0, device=device),
        torch.sin(theta_z), torch.cos(theta_z),  torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
    ]).reshape(3, 3)

    Rx = torch.stack([
        torch.tensor(1.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device), torch.cos(theta_x), -torch.sin(theta_x),
        torch.tensor(0.0, device=device), torch.sin(theta_x), torch.cos(theta_x)
    ]).reshape(3, 3)

    rotated_normal = Rz @ Rx @ normal
    return torch.norm(rotated_normal - target_normal)

###############################
for _ in range(100000):  
    optimizer.zero_grad()
    loss = loss_fn()
    loss.backward()
    optimizer.step()

theta_z_opt = theta_z.detach().cpu()
theta_x_opt = theta_x.detach().cpu()
print(f"Optimized Theta Z: {torch.rad2deg(theta_z_opt).item()} degrees")
print(f"Optimized Theta X: {torch.rad2deg(theta_x_opt).item()} degrees")

