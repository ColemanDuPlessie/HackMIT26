from pathlib import Path
import argparse
import numpy as np


# Each triplet is (point A, vertex B, point C).
# These indices assume the MediaPipe 33-landmark layout.
ANGLE_JOINTS = {
    "left_elbow": (11, 13, 15),
    "right_elbow": (12, 14, 16),
    "left_shoulder": (13, 11, 23),
    "right_shoulder": (14, 12, 24),
    "left_knee": (23, 25, 27),
    "right_knee": (24, 26, 28),
    "left_hip": (11, 23, 25),
    "right_hip": (12, 24, 26),
}

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def validate_poses(poses):
    """Require XYZ joint positions shaped (frames, 33, 3)."""
    poses = np.asarray(poses, dtype=float)

    if poses.ndim != 3 or poses.shape[1:] != (33, 3):
        raise ValueError(
            f"Expected (frames, 33, 3), received {poses.shape}. "
            "A different skeleton layout or vector format "
            "must be converted before scoring."
        )

    if len(poses) == 0:
        raise ValueError("The motion contains no frames.")

    # A joint is missing if any coordinate is NaN or infinite.
    poses = poses.copy()
    valid_joints = np.all(np.isfinite(poses), axis=2)
    poses[~valid_joints] = np.nan

    return poses


def validate_pair(reference, generated):
    reference = validate_poses(reference)
    generated = validate_poses(generated)

    if reference.shape != generated.shape:
        raise ValueError(
            f"Shapes differ: {reference.shape} vs {generated.shape}. "
            "The current scorer requires matching frame counts "
            "and corresponding frame timing."
        )

    return reference, generated


def calculate_angle(point_a, point_b, point_c):
    """Return the angle at point B in degrees."""
    points = np.asarray(
        [point_a, point_b, point_c],
        dtype=float,
    )

    if points.shape != (3, 3):
        raise ValueError("Each point must have three coordinates.")

    if not np.all(np.isfinite(points)):
        return np.nan

    vector_1 = points[0] - points[1]
    vector_2 = points[2] - points[1]

    length_1 = np.linalg.norm(vector_1)
    length_2 = np.linalg.norm(vector_2)

    if length_1 <= 1e-12 or length_2 <= 1e-12:
        return np.nan

    cosine = np.dot(
        vector_1 / length_1,
        vector_2 / length_2,
    )

    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def get_joint_angles(poses):
    """Return eight joint angles per frame."""
    poses = validate_poses(poses)

    angles = np.full(
        (len(poses), len(ANGLE_JOINTS)),
        np.nan,
    )

    for frame_index, frame in enumerate(poses):
        for angle_index, (a, b, c) in enumerate(
            ANGLE_JOINTS.values()
        ):
            angles[frame_index, angle_index] = calculate_angle(
                frame[a],
                frame[b],
                frame[c],
            )

    return angles


def angle_metrics(reference, generated):
    reference, generated = validate_pair(reference, generated)

    reference_angles = get_joint_angles(reference)
    generated_angles = get_joint_angles(generated)

    valid = (
        np.isfinite(reference_angles)
        & np.isfinite(generated_angles)
    )

    if not np.any(valid):
        raise ValueError("No valid matching angles were found.")

    errors = np.abs(
        reference_angles[valid] - generated_angles[valid]
    )

    mean_error = float(np.mean(errors))
    score = float(np.clip(1 - mean_error / 180, 0.0, 1.0))

    return {
        "score": score,
        "mean_error_degrees": mean_error,
        "coverage": float(np.mean(valid)),
    }


def angle_similarity(reference, generated):
    return angle_metrics(reference, generated)["score"]


def position_metrics(reference, generated, distance_scale=1.0):
    reference, generated = validate_pair(reference, generated)

    if not np.isfinite(distance_scale) or distance_scale <= 0:
        raise ValueError(
            "distance_scale must be finite and greater than zero."
        )

    joint_activity = np.zeros(reference.shape[1])

    # With one frame, movement cannot be measured.
    # In that case, every joint gets equal weight.
    if len(reference) > 1:
        reference_motion = np.diff(reference, axis=0)
        movement_amount = np.linalg.norm(reference_motion, axis=2)

        valid_motion = np.isfinite(movement_amount)

        totals = np.sum(
            np.where(valid_motion, movement_amount, 0.0),
            axis=0,
        )
        counts = np.sum(valid_motion, axis=0)

        np.divide(
            totals,
            counts,
            out=joint_activity,
            where=counts > 0,
        )

    maximum_activity = np.max(joint_activity)

    if maximum_activity > 0:
        joint_activity = joint_activity / maximum_activity

    # Weights range from 1 to 2.
    joint_weights = 1.0 + joint_activity

    distances = np.linalg.norm(reference - generated, axis=2)
    valid = np.isfinite(distances)

    if not np.any(valid):
        raise ValueError("No valid matching positions were found.")

    weights = np.broadcast_to(joint_weights, distances.shape)

    mean_distance = float(
        np.average(
            distances[valid],
            weights=weights[valid],
        )
    )

    # distance_scale uses the same units as the input coordinates.
    score = float(1 / (1 + mean_distance / distance_scale))

    return {
        "score": score,
        "mean_distance": mean_distance,
        "coverage": float(np.mean(valid)),
    }


def position_similarity(reference, generated, distance_scale=1.0):
    return position_metrics(
        reference,
        generated,
        distance_scale,
    )["score"]


def inspect_files(folder):
    """Print every NPZ filename and its array names and shapes."""
    folder = Path(folder)

    if not folder.is_dir():
        print(f"Examples folder does not exist: {folder}")
        return

    paths = sorted(folder.rglob("*.npz"))

    if not paths:
        print(f"No .npz files found in: {folder}")
        return

    print(f"Found {len(paths)} NPZ file(s).")

    for path in paths:
        print(f"\nFile: {path.name}")

        try:
            with np.load(path, allow_pickle=False) as data:
                if not data.files:
                    print("  This archive contains no arrays.")

                for key in data.files:
                    try:
                        array = data[key]
                        print(
                            f"  Key: {key!r}\n"
                            f"  Shape: {array.shape}\n"
                            f"  Type: {array.dtype}"
                        )
                    except ValueError as error:
                        print(f"  Key {key!r}: {error}")

        except (OSError, ValueError) as error:
            print(f"  Could not read file: {error}")


def load_motion(path, key=None):
    """Load a selected array without guessing among multiple keys."""
    path = Path(path)

    with np.load(path, allow_pickle=False) as data:
        if key is None:
            if len(data.files) != 1:
                raise ValueError(
                    f"{path.name} contains these keys: {data.files}. "
                    "Specify the appropriate --reference-key, "
                    "--best-key, or --worst-key."
                )

            key = data.files[0]

        if key not in data.files:
            raise ValueError(
                f"Key {key!r} not found in {path.name}. "
                f"Available keys: {data.files}"
            )

        poses = data[key].copy()

    return validate_poses(poses)


def evaluate(reference, generated, distance_scale):
    return {
        "angles": angle_metrics(reference, generated),
        "positions": position_metrics(
            reference,
            generated,
            distance_scale,
        ),
    }


def print_results(label, results):
    angles = results["angles"]
    positions = results["positions"]

    print(f"\n{label}")
    print(f"  Angle score:          {angles['score']:.4f}")
    print(
        f"  Mean angle error:     "
        f"{angles['mean_error_degrees']:.2f} degrees"
    )
    print(f"  Valid angle coverage: {angles['coverage']:.1%}")
    print(f"  Position score:       {positions['score']:.4f}")
    print(f"  Mean position error:  {positions['mean_distance']:.6f}")
    print(f"  Valid joint coverage: {positions['coverage']:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect NPZ files or compare dance motions."
    )

    parser.add_argument(
        "--examples",
        type=Path,
        default=EXAMPLES_DIR,
    )

    parser.add_argument("--reference", type=Path)
    parser.add_argument("--best", type=Path)
    parser.add_argument("--worst", type=Path)

    parser.add_argument("--reference-key")
    parser.add_argument("--best-key")
    parser.add_argument("--worst-key")

    parser.add_argument(
        "--distance-scale",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    paths = (args.reference, args.best, args.worst)

    # No file arguments: inspect the examples folder.
    if all(path is None for path in paths):
        inspect_files(args.examples)
        return

    if any(path is None for path in paths):
        parser.error(
            "Provide --reference, --best, and --worst together."
        )

    try:
        reference = load_motion(
            args.reference,
            args.reference_key,
        )
        best = load_motion(
            args.best,
            args.best_key,
        )
        worst = load_motion(
            args.worst,
            args.worst_key,
        )

        best_results = evaluate(
            reference,
            best,
            args.distance_scale,
        )
        worst_results = evaluate(
            reference,
            worst,
            args.distance_scale,
        )

        print_results("BEST MOTION", best_results)
        print_results("WORST MOTION", worst_results)

        angle_difference = (
            best_results["angles"]["score"]
            - worst_results["angles"]["score"]
        )
        position_difference = (
            best_results["positions"]["score"]
            - worst_results["positions"]["score"]
        )

        print("\nBEST MINUS WORST")
        print("  Positive means the best motion scored higher.")
        print(f"  Angle difference:    {angle_difference:+.4f}")
        print(f"  Position difference: {position_difference:+.4f}")

    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"\nError: {error}\n")


if __name__ == "__main__":
    main()
