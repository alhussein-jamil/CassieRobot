import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def symmetric_quaternion(q):
    # Convert quaternion to rotation matrix
    rot_matrix = R.from_quat(q).as_matrix()

    # Create reflection matrix for XOZ plane (reflect y)
    reflect_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])

    # Apply reflection
    np.dot(reflect_matrix, rot_matrix)

    # Convert back to quaternion
    symmetric_q = [q[0], -q[1], q[2], -q[3]]

    return symmetric_q


def visualize_quaternions(q1, q2):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Original vector (along x-axis)
    v = np.array([1, 0, 0])

    # Rotate vector using quaternions
    v1 = R.from_quat(q1).apply(v)
    v2 = R.from_quat(q2).apply(v)

    # Plot vectors
    ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color="r", label="Original")
    ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color="b", label="Symmetric")

    # Plot XOZ plane
    xx, zz = np.meshgrid(range(-1, 2), range(-1, 2))
    yy = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color="g")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Quaternion and its Symmetric along XOZ plane")

    plt.show()


# Example usage
original_q = R.from_euler("xyz", [30, 45, 60], degrees=True).as_quat()
symmetric_q = symmetric_quaternion(original_q)

print("Original quaternion:", original_q)
print("Symmetric quaternion:", symmetric_q)

visualize_quaternions(original_q, symmetric_q)
