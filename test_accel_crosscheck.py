"""
Proves the accelerometer cross-check (accel_sensor.py) actually does its
job: boost confidence when a real shock confirms an acoustic "impulsive"
classification, and REDUCE confidence when the mic hears something
impulsive-sounding but no physical shock occurred (the false-positive
case -- fireworks, a shout, backfire heard at a distance).

No real accelerometer hardware exists yet -- this uses MockAccelerometer
with scripted shock events, which is exactly what accel_sensor.py's mock
is for. This test is honest about that in its output.
"""

from accel_sensor import MockAccelerometer, cross_check_confidence


def main():
    print("=" * 60)
    print("Accelerometer cross-check test (using MockAccelerometer)")
    print("=" * 60)

    # Case 1: classifier says "impulsive", accelerometer independently
    # confirms a real shock at the same moment -> confidence should go UP.
    accel = MockAccelerometer()
    accel.script_shock_at(seconds_from_start=0.0, duration=1.0)  # "shock" happening right now
    acoustic_confidence = 0.65
    shock_score = accel.read_recent_shock_score()
    boosted = cross_check_confidence("impulsive", acoustic_confidence, shock_score)
    print(f"\nCase 1: real gunshot (mic AND accelerometer agree)")
    print(f"  acoustic confidence alone:      {acoustic_confidence:.2f}")
    print(f"  accelerometer shock score:      {shock_score:.2f}")
    print(f"  final confidence after cross-check: {boosted:.2f}  (should be HIGHER)")
    assert boosted > acoustic_confidence, "Expected confidence to increase when accelerometer confirms a shock"

    # Case 2: classifier says "impulsive" (e.g. fireworks heard on mic),
    # but the accelerometer sees NO physical shock -> confidence should
    # go DOWN. This is the actual false-positive fix.
    accel2 = MockAccelerometer()  # no scripted shocks -- nothing happened physically
    shock_score2 = accel2.read_recent_shock_score()
    penalized = cross_check_confidence("impulsive", acoustic_confidence, shock_score2)
    print(f"\nCase 2: likely false positive (mic thinks 'impulsive', no physical shock)")
    print(f"  acoustic confidence alone:      {acoustic_confidence:.2f}")
    print(f"  accelerometer shock score:      {shock_score2:.2f}")
    print(f"  final confidence after cross-check: {penalized:.2f}  (should be LOWER)")
    assert penalized < acoustic_confidence, "Expected confidence to decrease when accelerometer sees no shock"

    # Case 3: non-impulsive labels should be untouched by the accelerometer
    steady_confidence = 0.8
    unaffected = cross_check_confidence("steady", steady_confidence, shock_score2)
    print(f"\nCase 3: steady noise (accelerometer should have NO effect)")
    print(f"  confidence in:  {steady_confidence:.2f}")
    print(f"  confidence out: {unaffected:.2f}  (should be UNCHANGED)")
    assert unaffected == steady_confidence, "Expected non-impulsive labels to pass through untouched"

    print("\nAll cross-check behaviors verified correctly.")
    print("\nNOTE: MockAccelerometer was used -- this proves the LOGIC is")
    print("correct. It does not prove real hardware works. Swap in")
    print("ADXL345Accelerometer from accel_sensor.py the moment you have")
    print("a real sensor wired up over I2C, and re-run a version of this")
    print("test against real recorded shocks before trusting it live.")


if __name__ == "__main__":
    main()
