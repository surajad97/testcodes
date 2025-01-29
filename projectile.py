import math

# Constants
v0 = 50  # m/s
theta = math.radians(45)  # Launch angle (radians)
g = 9.81  # m/s²
m = 0.145  # kg (baseball mass)
Cd = 0.3  # Drag coefficient
Cl = 0.1  # Lift coefficient (adjust based on spin)
rho = 1.225  # kg/m³ (air density)
r = 0.01  # m (ball radius)
A = math.pi * r**2  # Cross-sectional area
dt = 0.01  # Time step (s)
wind_x = 5  # m/s (wind in x-direction)
spin_direction = 1  # 1 = backspin, -1 = topspin

# Initial conditions
x = 0.0  # m
y = 0.0  # m
vx = v0 * math.cos(theta)  # Initial vx (DO NOT ADD WIND HERE)
vy = v0 * math.sin(theta)  # Initial vy

while y >= 0:
    # Update position
    x += vx * dt
    y += vy * dt

    # Relative velocity (account for wind)
    v_rel_x = vx - wind_x
    v_rel_y = vy
    v_rel = math.sqrt(v_rel_x**2 + v_rel_y**2)

    # Drag force
    Fd_x = -0.5 * Cd * rho * A * v_rel * v_rel_x
    Fd_y = -0.5 * Cd * rho * A * v_rel * v_rel_y

    # Magnus force (perpendicular to velocity + spin)
    Fm_x = 0.5 * Cl * rho * A * v_rel * (-spin_direction * v_rel_y)
    Fm_y = 0.5 * Cl * rho * A * v_rel * (spin_direction * v_rel_x)

    # Acceleration
    dvx_dt = (Fd_x + Fm_x) / m
    dvy_dt = (Fd_y + Fm_y) / m - g

    # Update velocity
    vx += dvx_dt * dt
    vy += dvy_dt * dt

print(f"Horizontal distance: {x:.2f} m")