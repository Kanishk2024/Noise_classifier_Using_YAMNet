# Calibration: getting absolute dBA

Without calibration the numbers are **relative** (the tool treats 0 dBFS as 94 dB so they
look like decibels, but they are not real sound pressure levels). Phone and camera
microphones, auto-gain and noise suppression all change the result.

## What `calibration_db` means

`calibration_db` is an **offset in dB**:

    dBA = (A-weighted level of the file, in dBFS) + calibration_db

For a recording device that has not been calibrated, leave it `null`.

## Option A - you have an acoustic calibrator (best)

1. With the **same device and settings** you'll use on the road, record 10-20 s of the
   calibrator tone (typically 94 dB SPL @ 1 kHz). Keep gain/AGC settings fixed afterwards.
2. Trim the file to the steady tone only (no handling noise) and save it as `calib.wav`.
3. Run the analyser with it:

       python main.py road.mp4 --calib-file calibration/calib.wav --calib-ref-db 94

   The tool computes `offset = 94 - level_of_calib.wav_in_dBFS` and applies it.
4. To reuse it, the offset is logged (`... -> offset XX.XX dB`); put it in `config.yaml`
   under `noise.calibration_db` or pass `--calibration-db XX.XX`.

## Option B - compare with a reference meter

1. Record a steady noise (e.g. a fan or steady traffic) with your device while a trusted
   sound level meter / phone app reads, say, 68.0 dBA.
2. Run the analyser uncalibrated and note its LAeq (say 71.4 relative).
3. offset = reference - measured, i.e. `68.0 - (71.4 - 94.0) = 90.6`. Pass that as
   `--calibration-db 90.6`. (The relative scale already includes +94, so subtract it as shown.)

## Practical limits

- Calibration is only valid for the same device, gain settings and (roughly) the same
  frequency content. Treat results as indicative, not legally compliant measurements.
- Video files often have automatic gain control that distorts levels over time.
- The filter is accurate to ~0.5 dB up to 4 kHz at 16 kHz sampling; above that it rolls
  off differently from the ideal A-curve (the audio is resampled to 16 kHz for YAMNet).
- Clipping (flat-topped waveforms) invalidates level readings.
