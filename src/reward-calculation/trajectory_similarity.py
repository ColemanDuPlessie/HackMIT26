import numpy as np

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

ANGLE_JOINTS = {
    "left_elbow": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    "right_elbow": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),

    "left_shoulder": (LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP),
    "right_shoulder": (RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP),

    "left_knee": (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    "right_knee": (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),

    "left_hip": (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
    "right_hip": (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)
}

def calculate_angle(point_a, point_b, point_c):
    vector_1 = point_a - point_b
    vector_2 = point_c - point_b

    denominator = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)

    # Avoid dividing by zero if two joints overlap
    if denominator == 0:
        return np.nan

    cosine = np.dot(vector_1, vector_2) / denominator

    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.arccos(cosine)

    return np.degrees(angle)

def get_joint_angles(poses):

    all_angles = []

    # Go through every frame in the dance
    for frame in poses:

        frame_angles = []

        # Calculate each of our 8 joint angles
        for name, (a, b, c) in ANGLE_JOINTS.items():

            angle = calculate_angle(
                frame[a],
                frame[b],
                frame[c]
            )

            frame_angles.append(angle)

        all_angles.append(frame_angles)

    return np.array(all_angles)

def angle_similarity(reference, generated):

    reference_angles = get_joint_angles(reference)
    generated_angles = get_joint_angles(generated)

    # Difference between matching joint angles
    angle_errors = np.abs(reference_angles - generated_angles)

    # Average across all valid angles and frames
    mean_angle_error = np.nanmean(angle_errors)

    similarity = 1 - (mean_angle_error / 180)

    similarity = np.clip(similarity, 0.0, 1.0)

    print("Mean angle error:", mean_angle_error)
    print("Angle similarity:", similarity)

    return similarity

def trajectory_similarity(reference, generated):

    reference = np.asarray(reference, dtype=float)
    generated = np.asarray(generated, dtype=float)

    if reference.shape != generated.shape:
        raise ValueError(
            f"Shape mismatch: {reference.shape} vs {generated.shape}"
        )

    # Find how much each joint moves in the reference dance
    reference_motion = np.diff(reference, axis=0)
    movement_amount = np.linalg.norm(reference_motion, axis=2)
    joint_activity = np.mean(movement_amount, axis=0)

    # Scale activity from 0 to 1
    if np.max(joint_activity) > 0:
        normalized_activity = joint_activity / np.max(joint_activity)
    else:
        normalized_activity = np.zeros_like(joint_activity)

    # More active joints get more weight, but every joint still matters
    BASE_WEIGHT = 1.0
    ACTIVITY_WEIGHT = 1.0

    joint_weights = BASE_WEIGHT + ACTIVITY_WEIGHT * normalized_activity

    # Find the distance between matching joints
    distances = np.linalg.norm(reference - generated, axis=2)

    # Average the errors while accounting for joint importance
    weighted_mean_distance = np.average(
        distances,
        weights=np.broadcast_to(joint_weights, distances.shape)
    )

    # Temporary similarity scaling
    similarity = 1 / (1 + weighted_mean_distance)

    print("Joint activity:", joint_activity)
    print("Joint weights:", joint_weights)
    print("Weighted mean distance:", weighted_mean_distance)

    return similarity


# Testing
# if __name__ == "__main__":

#     reference = np.array([
#         [[0, 0, 0], [0.0, 1, 0], [0.0, 2, 0]],
#         [[0, 0, 0], [0.2, 1, 0], [0.4, 2, 0]],
#         [[0, 0, 0], [0.4, 1, 0], [0.8, 2, 0]],
#         [[0, 0, 0], [0.6, 1, 0], [1.2, 2, 0]]
#     ])

#     generated = np.array([
#         [[0, 0, 0], [0.0, 1, 0], [0.0, 2, 0]],
#         [[0, 0, 0], [0.2, 1, 0], [0.4, 2, 0]],
#         [[0, 0, 0], [0.4, 1, 0], [0.8, 2, 0]],
#         [[0, 0, 0], [0.6, 1, 0], [1.2, 2, 0]]
#     ])

#     # Test the angle system with one fake frame
#     test_pose = np.zeros((1, 33, 3))

#     reference_pose = np.zeros((1, 33, 3))
#     generated_pose = np.zeros((1, 33, 3))

#     reference_pose = np.zeros((1, 33, 3))

# # Shoulders
#     reference_pose[0, 11] = [0, 1, 2]
#     reference_pose[0, 12] = [0, -1, 2]

# # Elbows
#     reference_pose[0, 13] = [0, 2, 2]
#     reference_pose[0, 14] = [0, -2, 2]

# # Wrists
#     reference_pose[0, 15] = [1, 2, 2]
#     reference_pose[0, 16] = [1, -2, 2]

# # Hips
#     reference_pose[0, 23] = [0, 0.5, 1]
#     reference_pose[0, 24] = [0, -0.5, 1]

# # Knees
#     reference_pose[0, 25] = [0, 0.5, 0]
#     reference_pose[0, 26] = [0, -0.5, 0]

# # Ankles
#     reference_pose[0, 27] = [0, 0.5, -1]
#     reference_pose[0, 28] = [0, -0.5, -1]

#     generated_pose = reference_pose.copy()

#     angle_score = angle_similarity(
#     reference_pose,
#     generated_pose
# )

#     print("Final angle score:", angle_score)

#     score = trajectory_similarity(reference, generated)

#     print("Reference:", reference.shape)
#     print("Generated:", generated.shape)
#     print("Similarity score:", score)
