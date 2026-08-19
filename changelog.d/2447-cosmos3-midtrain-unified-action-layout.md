### Fixed

- **policies/cosmos3**: name every column of the DROID `midtrain` action. The layout listed 8 quaternion-form columns for the model's 10-wide unified action, so each name landed one or two columns early - a rotation component was emitted as `gripper` while the real grasp left as the unnamed `action_9` - and `action_mapping` accepted the misplaced `gripper` while refusing `grasp`. `midtrain` now names the unified action's columns (`tx,ty,tz,r0..r5,grasp`), as the other three embodiments already did.
