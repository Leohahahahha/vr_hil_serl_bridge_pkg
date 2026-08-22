#!/usr/bin/env python3
"""Deeper diagnostic: print EVERY non-invalid tracked device, no filtering.

Unlike diag_openvr.py (which skips devices that are neither connected nor
valid), this dumps all devices OpenVR knows about, plus the controller-role
mapping exactly like vr_ros2_node._update_controller_indices() does.

Run with SteamVR + vr_teleop.launch.py up:
    python3 diag_openvr_full.py
"""
import openvr


def main() -> None:
    try:
        vr = openvr.init(openvr.VRApplication_Background)
    except Exception as exc:
        print(f"[FAIL] openvr.init failed: {exc}")
        return

    try:
        print(f"OpenVR runtime version: {vr.getRuntimeVersion()}")
        poses = vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding,
            0.0,
            openvr.k_unMaxTrackedDeviceCount,
        )

        print("\n=== ALL non-invalid devices (no filtering) ===")
        print(f"{'idx':>4} {'class':<12} {'connected':<10} {'valid':<6} {'role':<16} {'serial'}")
        print("-" * 80)
        for i, pose in enumerate(poses):
            cls = vr.getTrackedDeviceClass(i)
            if cls == openvr.TrackedDeviceClass_Invalid:
                continue
            if cls == openvr.TrackedDeviceClass_HMD:
                cls_name, role = "HMD", "HMD"
                role2 = "HMD"
            elif cls == openvr.TrackedDeviceClass_Controller:
                r = vr.getControllerRoleForTrackedDeviceIndex(i)
                role2 = (
                    "controller-left" if r == openvr.TrackedControllerRole_LeftHand
                    else "controller-right" if r == openvr.TrackedControllerRole_RightHand
                    else "controller-?"
                )
                cls_name = "Controller"
                role = role2
            elif cls == openvr.TrackedDeviceClass_TrackingReference:
                cls_name, role = "BaseStation", "base"
                role2 = "base"
            elif cls == openvr.TrackedDeviceClass_GenericTracker:
                cls_name, role = "GenericTracker", ""
                role2 = ""
            else:
                cls_name, role = f"class{cls}", ""
                role2 = ""
            serial = ""
            try:
                serial = vr.getStringTrackedDeviceProperty(
                    i, openvr.Prop_SerialNumber_String
                )
            except Exception:
                serial = "?"
            print(
                f"{i:>4} {cls_name:<12} {str(pose.bDeviceIsConnected):<10} "
                f"{str(pose.bPoseIsValid):<6} {role2:<16} {serial}"
            )

        # Replicate _update_controller_indices exactly
        indices = {'left': None, 'right': None}
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if vr.getTrackedDeviceClass(i) != openvr.TrackedDeviceClass_Controller:
                continue
            r = vr.getControllerRoleForTrackedDeviceIndex(i)
            if r == openvr.TrackedControllerRole_LeftHand:
                indices['left'] = i
            elif r == openvr.TrackedControllerRole_RightHand:
                indices['right'] = i
        print(f"\n_node._controller_indices (left/right) = {indices}")

        if indices['right'] is not None:
            ok, state = vr.getControllerState(indices['right'])
            print(f"Right controller idx={indices['right']}: state_ok={ok}")
        else:
            print("No RIGHT controller index -> node never publishes")
    finally:
        openvr.shutdown()


if __name__ == "__main__":
    main()
