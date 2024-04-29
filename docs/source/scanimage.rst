ScanImage Tiff Files
~~~~~~~~~~~~~~~~~~~~

ScanImage records images, volumes and video as Tiff or BigTiff files.
For the most part Tiff and BigTiff are similar, but they are not the same;
BigTiff enables storage of larger (>4GB) data sets.  In addition to the image
data, ScanImage stores metadata describing the microscope configuration and
settings used during the acquisition.  These are stored in the file itself.

Some of this metadata is accessible using standard Tiff readers.  The `tiff
format`_ provides for data fields (tags) which can be used to attach data to
describing each frame.  Past versions of ScanImage would store a copy the
metadata in "image description" tag for each frame.  Much of the metadata is
redundant, and this can lead to longer load times and significant storage
overhead in some cases.

Fortunately, the tiff format is very flexible and allows us to easily store
the part of the metadata that doesn't change over time in a dedicated block
of the file. However, this metadata block is only accessible if you know where
to look. The tiff files can still be read by any reader conforming to the
baseline tiff specification, but those readers will not be aware of the ScanImage
metadata block.  The ScanImageTiffReader knows how to extract this metadata and
also provides fast access to the images themselves.

ScanImage tiff files store the following kinds of data:

.. table::

    ================ ===========================================================
    Kind             Description
    ================ ===========================================================
    **data**         The image data itself
    **metadata**     Frame-invariant metadata such as the microscope configuration_ and
                     `region of interest`_ descriptions.
    **descriptions** Frame-varying metadata such as timestamps_ and `I2C data`_.
    ================ ===========================================================

The metadata sections themselves are encoded as either matlab evaluable_
strings or json_.

.. _`tiff format`: https://partners.adobe.com/public/developer/en/tiff/TIFF6.pdf
.. _`I2C data`: http://scanimage.vidriotechnologies.com/display/SI2018/I2C+data+recording
.. _timestamps: http://scanimage.vidriotechnologies.com/display/SI2018/Auxiliary+Trigger
.. _`region of interest`: http://scanimage.vidriotechnologies.com/display/SI2018/Scanfields%2C+ROIs%2C+ROI+Groups
.. _configuration: http://scanimage.vidriotechnologies.com/display/SI2018/CFG+and+USR+Files
.. _evaluable: http://www.mathworks.com/help/matlab/ref/eval.html
.. _json: http://www.json.org/
.. _scanimage.org: http://scanimage.org