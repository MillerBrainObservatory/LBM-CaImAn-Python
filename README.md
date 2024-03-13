# Light Beads Microscopy (LBM) and 2pRAM Analysis Pipeline 

 Steps:
 - reshapes the axis to have an x,y,z,t volume (LBM)
 - sorts the z-planes (LBM)
 - calculates and corrects the MROI seams
 - calculates and corrects the X-Y shifts across planes (LBM)
 - outputs data as x-y-t planes or x-y-z-t volumes)

## Troubleshooting

If a silent crash happens (and the Terminal closes) despite not reaching 100% RAM,
it could be due to the system-oom-process-killer being too sensitive (or acting weird)
It might be worth trying to increase the thresholds for when to kill processes,
[or to turn it off altogether...](
https://askubuntu.com/questions/1404888/how-do-i-disable-the-systemd-oom-process-killer-in-ubuntu-22-04)


```bash
$ systemctl disable --now systemd-oomd
$ systemctl mask systemd-oomd
```
It can be turned back on with:
```
$ systemctl enable systemd-oomd
$ systemctl unmask systemd-oomd
```