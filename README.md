# Light Beads Microscopy (LBM) and 2pRAM Analysis Pipeline 


 ## Pre-processing pipeline of 2pRAM and LBM data 

 Steps:
 - reshapes the axis to have an x,y,z,t volume (LBM)
 - sorts the z-planes (LBM)
 - calculates and corrects the MROI seams
 - calculates and corrects the X-Y shifts across planes (LBM)
 - outputs data as x-y-t planes or x-y-z-t volumes)

## Troubleshooting

If a silent crash happens (and the Terminal closes) despite not reaching 100% RAM,
it could be due to the system-oom-process-killer being too sensitive (or acting weird)
It might be worth trying to increase the thresholds for when to kill processes, or to turn it off altogether...
https://askubuntu.com/questions/1404888/how-do-i-disable-the-systemd-oom-process-killer-in-ubuntu-22-04
To turned it off:

```bash
$ systemctl disable --now systemd-oomd
$ systemctl mask systemd-oomd
```
It can be turned back on with:
```
$ systemctl enable systemd-oomd
$ systemctl unmask systemd-oomd
```

### Steps:
- disentangles time and plane dimensions and reshapes to a 4D volume (LBM)
- sorts the z-planes (LBM)
- uses metadata to place MROIs and reconstruct the frames
- calculates and corrects the MROI overlaps and seams
- calculates and corrects the X-Y shifts across planes (LBM)
- outputs data as t-x-y planes or t-x-y-z volumes
- saves a .png with the average frame of all planes
- saves a short .mp4 clip
