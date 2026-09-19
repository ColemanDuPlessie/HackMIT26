import numpy as np
from pathlib import Path


# Z contributes 30% less than X and Y
AXIS_WEIGHTS = np.array([1.0, 1.0, 0.3])


# MediaPipe joint indices
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


# Calibration from the intentionally bad example
BAD_TARGET_SCORE = 0.30

# These were measured before Z downweighting.
# We should update position and trajectory values after rerunning.
POSITION_BAD_ERROR = 0.10640327391349057
ANGLE_BAD_ERROR = 10.483409009823326
TRAJECTORY_BAD_ERROR = 0.040978255094918876


POSITION_SCALE = (
    -POSITION_BAD_ERROR
    / np.log(BAD_TARGET_SCORE)
)

ANGLE_SCALE = (
    -ANGLE_BAD_ERROR
    / np.log(BAD_TARGET_SCORE)
)

TRAJECTORY_SCALE = (
    -TRAJECTORY_BAD_ERROR
    / np.log(BAD_TARGET_SCORE)
)


# Small errors are tolerated before the score starts dropping
POSITION_TOLERANCE = 0.03
ANGLE_TOLERANCE = 5.0
TRAJECTORY_TOLERANCE = 0.01


def weighted_xyz_norm(vectors):
    """
    Calculate XYZ distance while making Z count 30% less.
    """

    weighted_vectors = vectors * AXIS_WEIGHTS

    return np.linalg.norm(
        weighted_vectors,
        axis=-1
    )


def calculate_angle(point_a, point_b, point_c):
    vector_1 = point_a - point_b
    vector_2 = point_c - point_b

    denominator = (
        np.linalg.norm(vector_1)
        * np.linalg.norm(vector_2)
    )

    if denominator == 0:
        return np.nan

    cosine = (
        np.dot(vector_1, vector_2)
        / denominator
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    angle = np.arccos(cosine)

    return np.degrees(angle)


def get_joint_angles(poses):
    all_angles = []

    for frame in poses:
        frame_angles = []

        for name, (a, b, c) in ANGLE_JOINTS.items():

            angle = calculate_angle(
                frame[a],
                frame[b],
                frame[c]
            )

            frame_angles.append(angle)

        all_angles.append(frame_angles)

    return np.array(all_angles)


def get_joint_weights(reference):

    # Movement from one frame to the next
    reference_motion = np.diff(
        reference,
        axis=0
    )

    # Z is downweighted here too
    movement_amount = weighted_xyz_norm(
        reference_motion
    )

    joint_activity = np.mean(
        movement_amount,
        axis=0
    )

    if np.max(joint_activity) > 0:

        normalized_activity = (
            joint_activity
            / np.max(joint_activity)
        )

    else:

        normalized_activity = np.zeros_like(
            joint_activity
        )

    # Every joint matters, active joints matter more
    BASE_WEIGHT = 1.0
    ACTIVITY_WEIGHT = 1.0

    joint_weights = (
        BASE_WEIGHT
        + ACTIVITY_WEIGHT
        * normalized_activity
    )

    return joint_weights


def position_similarity(reference, generated):

    reference = np.asarray(
        reference,
        dtype=float
    )

    generated = np.asarray(
        generated,
        dtype=float
    )

    if reference.shape != generated.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"{reference.shape} vs {generated.shape}"
        )

    joint_weights = get_joint_weights(
        reference
    )

    differences = (
        reference - generated
    )

    # X and Y full strength, Z at 70%
    distances = weighted_xyz_norm(
        differences
    )

    position_error = np.average(
        distances,
        weights=np.broadcast_to(
            joint_weights,
            distances.shape
        )
    )

    effective_error = max(
        0.0,
        position_error
        - POSITION_TOLERANCE
    )

    similarity = np.exp(
        -effective_error
        / POSITION_SCALE
    )

    return (
        similarity,
        position_error
    )


def angle_similarity(reference, generated):

    reference_angles = get_joint_angles(
        reference
    )

    generated_angles = get_joint_angles(
        generated
    )

    angle_errors = np.abs(
        reference_angles
        - generated_angles
    )

    angle_error = np.nanmean(
        angle_errors
    )

    effective_error = max(
        0.0,
        angle_error
        - ANGLE_TOLERANCE
    )

    similarity = np.exp(
        -effective_error
        / ANGLE_SCALE
    )

    return (
        similarity,
        angle_error
    )


def trajectory_similarity(reference, generated):

    reference = np.asarray(
        reference,
        dtype=float
    )

    generated = np.asarray(
        generated,
        dtype=float
    )

    if reference.shape != generated.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"{reference.shape} vs {generated.shape}"
        )

    reference_motion = np.diff(
        reference,
        axis=0
    )

    generated_motion = np.diff(
        generated,
        axis=0
    )

    motion_difference = (
        reference_motion
        - generated_motion
    )

    # Z movement contributes 30% less
    motion_errors = weighted_xyz_norm(
        motion_difference
    )

    joint_weights = get_joint_weights(
        reference
    )

    trajectory_error = np.average(
        motion_errors,
        weights=np.broadcast_to(
            joint_weights,
            motion_errors.shape
        )
    )

    effective_error = max(
        0.0,
        trajectory_error
        - TRAJECTORY_TOLERANCE
    )

    similarity = np.exp(
        -effective_error
        / TRAJECTORY_SCALE
    )

    return (
        similarity,
        trajectory_error
    )


def align_timing(
    reference,
    generated,
    fps,
    max_shift_seconds=1.0
):

    reference = np.asarray(
        reference,
        dtype=float
    )

    generated = np.asarray(
        generated,
        dtype=float
    )

    max_shift_frames = int(
        max_shift_seconds * fps
    )

    max_shift_frames = min(
        max_shift_frames,
        len(reference) - 1,
        len(generated) - 1
    )

    joint_weights = get_joint_weights(
        reference
    )

    best_error = np.inf
    best_shift = 0

    best_reference = reference
    best_generated = generated

    # Try moving the player earlier/later
    for shift in range(
        -max_shift_frames,
        max_shift_frames + 1
    ):

        if shift > 0:

            ref_part = reference[:-shift]
            gen_part = generated[shift:]

        elif shift < 0:

            amount = -shift

            ref_part = reference[amount:]
            gen_part = generated[:-amount]

        else:

            ref_part = reference
            gen_part = generated

        differences = (
            ref_part - gen_part
        )

        # Z matters less when finding best timing alignment too
        distances = weighted_xyz_norm(
            differences
        )

        error = np.average(
            distances,
            weights=np.broadcast_to(
                joint_weights,
                distances.shape
            )
        )

        if error < best_error:

            best_error = error
            best_shift = shift

            best_reference = ref_part
            best_generated = gen_part

    timing_error_seconds = (
        abs(best_shift) / fps
    )

    # Still temporary until timing is calibrated
    timing_similarity = (
        1
        / (1 + timing_error_seconds)
    )

    return (
        best_reference,
        best_generated,
        timing_similarity,
        best_shift,
        timing_error_seconds
    )


def dance_similarity(
    reference,
    generated,
    fps
):

    (
        aligned_reference,
        aligned_generated,
        timing_score,
        shift,
        timing_error
    ) = align_timing(
        reference,
        generated,
        fps
    )

    (
        position_score,
        position_error
    ) = position_similarity(
        aligned_reference,
        aligned_generated
    )

    (
        angle_score,
        angle_error
    ) = angle_similarity(
        aligned_reference,
        aligned_generated
    )

    (
        trajectory_score,
        trajectory_error
    ) = trajectory_similarity(
        aligned_reference,
        aligned_generated
    )

    # Equal weighting for now
    position_weight = 0.25
    angle_weight = 0.25
    trajectory_weight = 0.25
    timing_weight = 0.25

    # Weighted geometric mean
    final_score = (
        position_score ** position_weight
        * angle_score ** angle_weight
        * trajectory_score ** trajectory_weight
        * timing_score ** timing_weight
    )

    return {
        "final_score":
            final_score,

        "position_score":
            position_score,

        "position_error":
            position_error,

        "angle_score":
            angle_score,

        "angle_error_degrees":
            angle_error,

        "trajectory_score":
            trajectory_score,

        "trajectory_error":
            trajectory_error,

        "timing_score":
            timing_score,

        "timing_shift_frames":
            shift,

        "timing_error_seconds":
            timing_error
    }


if __name__ == "__main__":

    examples_dir = (
        Path(__file__).resolve().parents[1]
        / "examples"
    )

    best_data = np.load(
        examples_dir
        / "danceOne_best_keypoints.npz"
    )

    worst_data = np.load(
        examples_dir
        / "danceOne_worst_keypoints.npz"
    )

    reference = best_data["world"]
    worst = worst_data["world"]

    fps = float(
        best_data["fps"]
    )

    print(
        "Reference shape:",
        reference.shape
    )

    print(
        "Worst shape:",
        worst.shape
    )

    print(
        "FPS:",
        fps
    )

    print(
        "XYZ weights:",
        AXIS_WEIGHTS
    )


    # Perfect reference test
    perfect_result = dance_similarity(
        reference,
        reference,
        fps
    )

    print("\nBEST VS BEST")

    for key, value in perfect_result.items():
        print(
            key,
            ":",
            value
        )


    # Bad motion test
    worst_result = dance_similarity(
        reference,
        worst,
        fps
    )

    print("\nBEST VS WORST")

    for key, value in worst_result.items():
        print(
            key,
            ":",
            value
        )
