#!/usr/bin/env python3
"""Diagnose which OpenVR tracked devices are connected & being tracked.

Run AFTER SteamVR + vr_teleop.launch.py are up:
    python3 diag_openvr.py

It prints, for each tracked device: class, connected, valid pose, and role
(controller left/right / HMD). This tells us whether the problem is the HMD
not being tracked (first return in _timer_cb) or the right controller
(second return).
"""
import openvr


def main() -> None:
    try:
        vr = openvr.init(openvr.VRApplication_Background)
    except Exception as exc:
        print(f"[FAIL] openvr.init failed: {exc}")
        print("  -> SteamVR does not appear to be running / reachable.")
        return

    try:
        print(f"OpenVR runtime version: {vr.getRuntimeVersion()}")
        poses = vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding,
            0.0,
            openvr.k_unMaxTrackedDeviceCount,
        )
        print(f"\n{'idx':>4} {'class':<12} {'connected':<10} {'valid':<6} {'role'}")
        print("-" * 60)
        for i, pose in enumerate(poses):
            cls = vr.getTrackedDeviceClass(i)
            if cls == openvr.TrackedDeviceClass_Invalid:
                continue
            if not (pose.bDeviceIsConnected or pose.bPoseIsValid):
                continue

            if cls == openvr.TrackedDeviceClass_HMD:
                cls_name, role = "HMD", "HMD"
            elif cls == openvr.TrackedDeviceClass_Controller:
                r = vr.getControllerRoleForTrackedDeviceIndex(i)
                role = (
                    "controller-left" if r == openvr.TrackedControllerRole_LeftHand
                    else "controller-right" if r == openvr.TrackedControllerRole_RightHand
                    else "controller-?"
                )
                cls_name = "Controller"
            elif cls == openvr.TrackedDeviceClass_TrackingReference:
                cls_name, role = "BaseStation", "base"
            else:
                cls_name, role = f"class{cls}", ""
            print(
                f"{i:>4} {cls_name:<12} {str(pose.bDeviceIsConnected):<10} "
                f"{str(pose.bPoseIsValid):<6} {role}"
            )

        # For the right controller, also report live pose once.
        for i, pose in enumerate(poses):
            if vr.getTrackedDeviceClass(i) != openvr.TrackedDeviceClass_Controller:
                continue
            if vr.getControllerRoleForTrackedDeviceIndex(i) != openvr.TrackedControllerRole_RightHand:
                continue
            ok, state = vr.getControllerState(i)
            trigger = state.rAxis[1].x if len(state.rAxis) > 1 else 0.0
            print(f"\nRight controller idx={i}: getControllerState={ok} trigger={trigger:.3f}")
    finally:
        openvr.shutdown()


if __name__ == "__main__":
    main()
