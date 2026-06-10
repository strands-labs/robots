"""AWS IoT Core integration for the strands-robots mesh.

This subpackage owns the cloud-side concerns of the mesh:

- :mod:`provision`  — single-Thing bootstrap (cert + policy + Thing).
- :mod:`bootstrap`  — account-wide bootstrap (Rules + Lambda +
  DynamoDB audit + Fleet Provisioning template).
- :mod:`shadow`  — Device Shadow named-shadow mirror of presence.
- :mod:`camera_offload` — S3-backed camera frame offload.

The wire-level transport (:class:`IotMqttTransport`) lives in
:mod:`strands_robots.mesh.transport.iot_transport` and is independent of
this package — you can use it without ever calling :mod:`provision` if you
already have certs.

Public OOBE
-----------
For a customer's first integration, the canonical sequence is::

    from strands_robots.mesh.iot import bootstrap_account, provision_robot

    # 1. Once per AWS account/region: spin up Rules / Lambda / DynamoDB /
    #  Fleet Provisioning template / etc.
    bootstrap_account()

    # 2. Per robot: issue a cert + attach the strands-robot policy.
    p = provision_robot("so100-arm-01")

    # 3. Run the robot under the iot transport.
    #  (export the env vars from p.env_vars() and `Robot()` Just Works.)

For a single-robot dev setup, step 1 is optional — without it E-stop
fan-out and DynamoDB audit just don't activate; everything else still
works.
"""

from strands_robots.mesh.iot.bootstrap import (
    BootstrappedAccount,
    bootstrap_account,
    teardown_account,
)
from strands_robots.mesh.iot.camera_offload import CameraOffloader
from strands_robots.mesh.iot.camera_offload import enable_for_mesh as enable_camera_offload_for_mesh
from strands_robots.mesh.iot.provision import (
    ProvisionedThing,
    provision_operator,
    provision_robot,
    teardown_thing,
)
from strands_robots.mesh.iot.shadow import (
    ShadowMirror,
    shadow_get_topic,
    shadow_update_topic,
)
from strands_robots.mesh.iot.shadow import enable_for_mesh as enable_shadow_for_mesh

__all__ = [
    # Provision
    "ProvisionedThing",
    "provision_robot",
    "provision_operator",
    "teardown_thing",
    # Bootstrap
    "BootstrappedAccount",
    "bootstrap_account",
    "teardown_account",
    # Shadow
    "ShadowMirror",
    "shadow_update_topic",
    "shadow_get_topic",
    "enable_shadow_for_mesh",
    # Camera
    "CameraOffloader",
    "enable_camera_offload_for_mesh",
]
