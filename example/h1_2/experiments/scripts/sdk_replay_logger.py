#!/usr/bin/env python3
"""sdk_replay_logger.py - Block 2 cmd-vs-exe logger over raw DDS (no ROS2).

Primary Block 2 logger since 2026-07-13. That bring-up produced two
all-NaN logs from the ROS2 replay_logger; root cause was ROS2's automatic
rt/ topic-name prefix (subscribing "rt/lowstate" from ROS2 listens on DDS
rt/rt/lowstate, which nobody publishes). The ROS2 node is fixed too, but
this logger subscribes with unitree_sdk2py on the exact same DDS stack as
replay_arm.py, so if the replay works, logging works; no ROS sourcing, no
name mapping.

Identical CSV to the ROS2 logger (TrialWriter and joint order are imported
from the h12_experiments package): t_wall, t_rel, weight, mode_machine,
then cmd_/exe_/err_<joint> for the 15 arm/waist joints. success_check.py
consumes it unchanged.

Guard rails the first bring-up lacked:
  - exits nonzero after --wait s if rt/lowstate never arrives (no more
    silent 90 s all-NaN logs); the useless CSV is removed
  - live status line every 5 s with lowstate/cmd message counters
  - summary warns if zero rt/arm_sdk messages were seen (replay not
    running, or not visible from this process)

Run (same conda env as replay_arm.py, e.g. rical_unitree; NO ROS needed):
    python3 sdk_replay_logger.py --interface enp128s31f6 \
        --motion 1 --trial 1 --duration 90 \
        --out-dir ~/mj_ws/h1-2_sensors/experiments/iros2026ws/block2

Offline end-to-end check (real DDS round-trip on loopback, no robot):
    python3 sdk_replay_logger.py --self-test
"""

import argparse
import math
import os
import sys
import threading
import time

PKG = os.path.expanduser("~/mj_ws/h1-2_sensors/yolo_ws/src/h12_experiments")
sys.path.insert(0, PKG)
from h12_experiments.joints import (  # noqa: E402
    LOG_JOINTS, LOG_INDEX, ARM_SDK_WEIGHT_INDEX)
from h12_experiments.replay_logger import TrialWriter  # noqa: E402


def run(args):
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize, ChannelSubscriber)
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

    if args.interface:
        ChannelFactoryInitialize(0, args.interface)
    else:
        ChannelFactoryInitialize(0)

    rep = f"_R{args.rep}" if args.rep else ""
    out_path = os.path.join(
        os.path.expanduser(args.out_dir),
        f"replay_M{args.motion}{rep}_T{args.trial}.csv")
    if os.path.exists(out_path):
        sys.exit(f"[sdk-logger] REFUSING to overwrite existing {out_path}. "
                 "Bump --trial (or --rep); no trial data is lost this way.")
    writer = TrialWriter(out_path, args.rate)
    counts = {"state": 0, "cmd": 0}

    def on_state(msg):
        counts["state"] += 1
        writer.set_exe([msg.motor_state[i].q for i in LOG_INDEX],
                       msg.mode_machine,
                       [msg.motor_state[i].dq for i in LOG_INDEX],
                       [msg.motor_state[i].tau_est for i in LOG_INDEX])

    def on_cmd(msg):
        counts["cmd"] += 1
        writer.set_cmd([msg.motor_cmd[i].q for i in LOG_INDEX],
                       msg.motor_cmd[ARM_SDK_WEIGHT_INDEX].q,
                       [msg.motor_cmd[i].tau for i in LOG_INDEX])

    sub_state = ChannelSubscriber("rt/lowstate", LowState_)
    sub_state.Init(on_state, 10)
    sub_cmd = ChannelSubscriber(args.cmd_topic, LowCmd_)
    sub_cmd.Init(on_cmd, 10)

    print(f"[sdk-logger] waiting up to {args.wait:.0f} s for rt/lowstate "
          f"(interface: {args.interface or 'default'})...")
    t0 = time.time()
    while counts["state"] == 0:
        if time.time() - t0 > args.wait:
            writer.close()
            os.remove(out_path)
            sys.exit("[sdk-logger] FATAL: no rt/lowstate. Robot up? Right "
                     "--interface? Same one replay_arm.py uses. Nothing "
                     "was logged; the empty CSV was removed.")
        time.sleep(0.05)
    print(f"[sdk-logger] rt/lowstate alive; logging @ {args.rate:g} Hz "
          f"-> {out_path}")
    print(f"[sdk-logger] duration: "
          f"{args.duration and f'{args.duration:g} s' or 'until Ctrl+C'}")

    dt = 1.0 / args.rate
    next_t = time.time()
    last_status = time.time()
    try:
        while True:
            writer.tick()
            if args.duration and writer.n * dt >= args.duration:
                print(f"[sdk-logger] reached duration {args.duration:g} s")
                break
            now = time.time()
            if now - last_status >= 5.0:
                print(f"[sdk-logger] t={writer.n * dt:6.1f} s  rows={writer.n}"
                      f"  lowstate={counts['state']}  cmd={counts['cmd']}")
                last_status = now
            next_t += dt
            time.sleep(max(0.0, next_t - time.time()))
    except KeyboardInterrupt:
        print("\n[sdk-logger] Ctrl+C, flushing")
    finally:
        writer.close()
        writer.summary()
        if counts["cmd"] == 0:
            print(f"[sdk-logger] WARNING: zero {args.cmd_topic} messages: "
                  "replay_arm.py never ran, or is on another interface.")
    return out_path


# ── self-test: full DDS round-trip on loopback, no robot ─────────────────────
def self_test(args):
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize, ChannelPublisher)
    from unitree_sdk2py.idl.default import (
        unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_)
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

    ChannelFactoryInitialize(0, args.interface or "lo")
    args.interface = None          # factory is already initialized
    pub_state = ChannelPublisher("rt/lowstate", LowState_)
    pub_state.Init()
    pub_cmd = ChannelPublisher(args.cmd_topic, LowCmd_)
    pub_cmd.Init()
    stop = threading.Event()

    def feeder():
        st = unitree_hg_msg_dds__LowState_()
        st.mode_machine = 4
        cm = unitree_hg_msg_dds__LowCmd_()
        cm.motor_cmd[ARM_SDK_WEIGHT_INDEX].q = 1.0
        t0 = time.time()
        while not stop.is_set():
            t = time.time() - t0
            for k, i in enumerate(LOG_INDEX):
                q = 0.4 * math.sin(2 * math.pi * 0.5 * t + 0.2 * k)
                cm.motor_cmd[i].q = q
                cm.motor_cmd[i].tau = 2.0 * math.cos(2 * math.pi * 0.5 * t)
                st.motor_state[i].q = 0.9 * q
                st.motor_state[i].dq = 0.9 * 0.4 * 2 * math.pi * 0.5 * math.cos(
                    2 * math.pi * 0.5 * t + 0.2 * k)
                st.motor_state[i].tau_est = 5.0 * math.sin(2 * math.pi * 0.5 * t)
            pub_state.Write(st)
            pub_cmd.Write(cm)
            time.sleep(0.005)

    th = threading.Thread(target=feeder, daemon=True)
    th.start()
    args.duration = args.duration or 3.0
    args.out_dir = args.out_dir_selftest
    print("[self-test] loopback publishers up; running the real logger")
    out_path = run(args)
    stop.set()
    th.join(timeout=1.0)

    import csv
    rows = list(csv.DictReader(open(out_path)))
    paired = [r for r in rows
              if r["cmd_left_elbow"] != "nan" and r["exe_left_elbow"] != "nan"]
    frac = len(paired) / max(1, len(rows))
    print(f"[self-test] {len(rows)} rows, {len(paired)} paired "
          f"({100 * frac:.0f}%)")
    if frac < 0.8:
        sys.exit("[self-test] FAIL: expected >= 80% paired cmd+exe rows")
    err = [abs(float(r["err_right_elbow"])) for r in paired]
    print(f"[self-test] err_right_elbow mean {sum(err) / len(err):.4f} rad "
          "(expect ~10% of a 0.4 rad sine)")
    tau_ok = [r for r in paired if r.get("cmdtau_right_elbow", "nan") != "nan"]
    if len(tau_ok) < 0.8 * len(paired):
        sys.exit("[self-test] FAIL: commanded tau not logged (cmdtau_ column)")
    tmax = max(abs(float(r["cmdtau_right_elbow"])) for r in tau_ok)
    print(f"[self-test] cmdtau_right_elbow logged, peak {tmax:.2f} Nm "
          "(expect ~2.0)")
    for col, peak_want in (("exedq_right_elbow", 0.9 * 0.4 * math.pi),
                           ("exetau_right_elbow", 5.0)):
        ok = [r for r in paired if r.get(col, "nan") != "nan"]
        if len(ok) < 0.8 * len(paired):
            sys.exit(f"[self-test] FAIL: {col} not logged (energy columns)")
        pk = max(abs(float(r[col])) for r in ok)
        print(f"[self-test] {col} logged, peak {pk:.2f} (expect ~{peak_want:.2f})")
    print("[self-test] PASS: DDS subscribe -> CSV path works end to end "
          "(incl. measured-energy columns)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interface", default=None,
                   help="robot network interface, e.g. enp128s31f6 "
                        "(must match replay_arm.py)")
    p.add_argument("--motion", type=int, default=0)
    p.add_argument("--rep", type=int, default=0,
                   help="reference-rep index; adds _R<rep> to the filename so "
                        "replaying M2_1/M2_2/M2_3 stays distinguishable (0 = "
                        "omit). Filenames: replay_M<motion>_R<rep>_T<trial>.csv")
    p.add_argument("--trial", type=int, default=0)
    p.add_argument("--rate", type=float, default=50.0)
    p.add_argument("--duration", type=float, default=0.0,
                   help="seconds; 0 = until Ctrl+C")
    p.add_argument("--wait", type=float, default=10.0,
                   help="abort if no rt/lowstate within this many s")
    p.add_argument("--out-dir", default=os.path.expanduser(
        "~/mj_ws/h1-2_sensors/experiments/iros2026ws/block2"))
    p.add_argument("--cmd-topic", default="rt/arm_sdk")
    p.add_argument("--self-test", action="store_true",
                   help="loopback DDS round-trip, no robot needed")
    args = p.parse_args()
    args.out_dir_selftest = "/tmp"

    if args.self_test:
        import tempfile
        args.out_dir_selftest = tempfile.mkdtemp(prefix="sdk_logger_test_")
        self_test(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
