### Fixed: without MuJoCo, a model that attaches another model keeps the attached joints and their limits

The MJCF walk used when the `mujoco` extra is not installed skipped `<attach>`, so LeKiwi read as three wheels with
no arm: its six arm joints, their actuators and their limits were missing. The walk now reads an attached model where
it is attached, with that model's own `<compiler>` settings and the attach prefix, and finds the attached file next to
the file that names it, as MuJoCo does. LeKiwi now gives the same joints, actuators and limits with and without MuJoCo.
An actuator whose joint the walk cannot find now refuses the model with a hint to install `mujoco`, instead of
leaving that joint with no limit.
