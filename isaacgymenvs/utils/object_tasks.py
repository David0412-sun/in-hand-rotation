"""Object selection and stable output directories shared by training entry points."""

import os

from omegaconf import OmegaConf


OBJECT_SETS = {
    "ball": ["ball"],
    "custom": ["box_50x50x50"],
    "cross": ["cross4_0", "cross4_1", "cross4_2", "cross4_3", "cross4_4"],
    "C": [
        "set_obj1_regular_block", "set_obj2_block", "set_obj3_block",
        "set_obj4_block", "set_obj5_block", "set_obj6_block_corner",
        "set_obj7_block", "set_obj8_short_block", "set_obj9_thin_block",
        "set_obj10_thin_block_corner", "set_obj11_cylinder",
        "set_obj12_cylinder_corner", "set_obj13_irregular_block",
        "set_obj14_irregular_block_cross", "set_obj15_irregular_block_time",
        "set_obj16_cylinder_axis",
    ],
}


def select_objects(object_set, object_name=""):
    """An empty name preserves the original training over the entire set."""
    objects = OBJECT_SETS[str(object_set)]
    if object_name:
        if object_name not in objects:
            raise ValueError(
                "Object {!r} is not in set {!r}; choose from {}".format(
                    object_name, object_set, ", ".join(objects)
                )
            )
        return [object_name]
    return list(objects)


def configure_output_directory(cfg, distillation=False):
    """Route by object and task axis, independently of experiment parameters."""
    env = cfg.task.env
    objects = select_objects(env.objSet, env.get("objectName", ""))
    axis = str(env.axis)
    if axis not in ("x", "y", "z"):
        raise ValueError("Rotation axis must be x, y, or z")

    if len(objects) == 1:
        object_dir = objects[0]
        model_name = objects[0]
    else:
        # A policy trained on a set does not belong to any one object.
        object_dir = os.path.join("_object_sets", str(env.objSet))
        model_name = str(env.objSet)
    task_dir = os.path.join(object_dir, axis + "-axis")

    config = cfg.train.params.config
    config.name = model_name + "_" + axis
    config.prefix = ""
    OmegaConf.update(config, "full_experiment_name", task_dir, force_add=True)
    output_dir = os.path.join(config.get("train_dir", "runs"), task_dir)
    if distillation:
        stages = {"warmup": "student", "collect": "collection"}
        if cfg.distill.bc_training not in stages:
            raise ValueError("distill.bc_training must be warmup or collect")
        stage = stages[cfg.distill.bc_training]
        output_dir = os.path.join(output_dir, stage)
        cfg.distill.student_logdir = output_dir
    return output_dir
