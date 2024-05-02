Light Beads Microscopy (LBM) and 2pRAM Analysis Pipeline
========================================================

The **Light Beads Microscopy (LBM)** and **2pRAM Analysis Pipeline** are designed to process and analyze microscopy data with several automated steps to enhance, classify, and correct the data acquired through advanced microscopy techniques.

Pipeline Steps
--------------
- **Reshapes the axis** to have an x, y, z, t volume (LBM).
- **Sorts the z-planes** (LBM).
- **Calculates and corrects the MROI seams**.
- **Calculates and corrects the X-Y shifts** across planes (LBM).
- **Outputs data** as x-y-t planes or x-y-z-t volumes.

This pipeline integrates functions such as :func:`classify_manual` which opens a GUI for manually classifying masks against a template image, enhancing the usability and precision of data classification in microscopy studies.

Troubleshooting
---------------
If a silent crash occurs (and the Terminal closes) despite not reaching 100% RAM, it could be due to the system-oom-process-killer being too sensitive or malfunctioning. Consider adjusting the thresholds for when processes are terminated, or temporarily disabling the system-oom-process-killer.

.. code-block:: bash

    $ systemctl disable --now systemd-oomd
    $ systemctl mask systemd-oomd

To reactivate the service:

.. code-block:: bash

    $ systemctl enable --now systemd-oomd
    $ systemctl unmask systemd-oomd
