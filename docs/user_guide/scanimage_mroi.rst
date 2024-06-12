Roi Hierarchy
=============

ScanImage® organizes Rois and Scanfields in a hierarchy. The root of this hierarchy is a RoiGroup object. Each RoiGroup object can contain multiple Rois, and each Roi can contain multiple scanfields.

- RoiGroup
  
  Roi
  
  Scanfield, z = 0
  Scanfield, z = 0.2
  Scanfield, z = 0.3
  Roi
  
  Scanfield, z = 0.1
  Scanfield, z = 0.5
  Roi
  
  Scanfield, z = 0.4
  Scanfield, z = 1

RoiGroup¶
=========

A RoiGroup is a container that holds multiple Rois. There are multiple accessor methods to manipulate the Rois within a RoiGroup. The order of the Rois within the RoiGroup determines the order the Rois that are scanned on one slice.


RoiGroup, Roi, and Scanfield are handle objects. The same Roi can be added to a RoiGroup multiple times, or be used in different RoiGroups. The same Scanfield can be added to different Rois, or multiple times to a single Roi.

.. code-block:: matlab

  % insert Roi at defined position
  hRoi0 = scanimage.mroi.Roi();             % create an empty Roi
  hRoiGroup.insertAfterId(0,hRoi0);         % add hRoi0 to beginning of hRoiGroup

  % reorder Roi list
  newIdx = hRoiGroup.moveById(2,1);         % move second Roi one step towards end
  newIdx = hRoiGroup.moveById(2,-1);        % move second Roi one step towards front
  newIdx = hRoiGroup.moveToFrontById(2);    % move second Roi to front
  newIdx = hRoiGroup.moveToBackById(2);     % move second Roi to back

  hRoi = hRoiGroup.removeById(2);           % removes second Roi from hRoiGroup

.. note::

  Each Roi, RoiGroup, and Scanfield object has a unique identifier ‘uuid’, which is a human-readable string. For performance reasons, a 64-bit integer value ‘uuiduint64’ is derived from ‘uuid’, which can be used to identify the object in the RoiGroup.
  Manipulate RoiGroup using uuiduint64


.. code-block:: Matlab

    hMyRoi = scanimage.mroi.Roi();                         % create an empty Roi
    myRoi_uuid = hMyRoi.uuid;                              % get unique identifier (human readble string)
    disp(hMyRoi.uuid);                                     % display human readable unique identifier
    myRoi_uuiduint64 = hMyRoi.uuiduint64;                  % get unique identifier (uint64)


    uuiduint64s = [hRoiGroup.rois.uuiduint64];             % get uuiduint64s for all Rois in hRoiGroup
    hRoiGroup.insertAfterId(uuiduint64s(2),hMyRoi);

    % manipulate RoiGroup using uuid - this is slow!!!
    newIdx = hRoiGroup.moveById(myRoi_uuid,1);             % this works, but is slow; use uuiduint64 instead for better performance!

    % manipulate RoiGroup using uuiduint64
    newIdx = hRoiGroup.moveById(myRoi_uuiduint64,1)        % move hMyRoi one step towards end
    newIdx = hRoiGroup.moveById(myRoi_uuiduint64,-1)       % move hMyRoi one step towards front
    newIdx = hRoiGroup.moveToFrontById(myRoi_uuiduint64)   % move hMyRoi to front
    newIdx = hRoiGroup.moveToBackById(myRoi_uuiduint64)    % move hMyRoi to back

    idx = hRoiGroup.idToIndex(roi_uuiduint64);             % find hMyRoi in hRoiGroup
    hMyRoi = hRoiGroup(idx);                               % get hMyRoi from hRoiGroup

    hMyRoi = hRoiGroup.removeById(roi_uuiduint64);         % removes hMyRoi from hRoiGroup

Roi
===

¶
Manipulate Rois
---------------

.. code-block:: matlab

    hSf0 = scanimage.mroi.scanfield.fields.RotatedRectangle();  % create an imaging Scanfield
    hSf1 = scanimage.mroi.scanfield.fields.RotatedRectangle();  % create an imaging Scanfield
    hSf2 = scanimage.mroi.scanfield.fields.RotatedRectangle();  % create an imaging Scanfield

    hRoi = scanimage.mroi.Roi();                                % create an empty Roi

    hRoi.add(0,hSf0);                                           % add Scanfield at z = 0
    hRoi.add(1,hSf1);                                           % add Scanfield at z = 1
    hRoi.add(2,hSf2);                                           % add Scanfield at z = 2

    hSf.scanfields                                              % show lists of Scanfields in roi
    hSf.zs                                                      % show list of zs

    tf = hSf.hit(3)                                             % check if Roi is defined at z=3

    tf = hSf.hit(0.5)                                           % check if Roi is defined at z=0.5
    hSf_interpolated = hRoi.get(0.5);                           % return interpolated Scanfield at z=0.5

    idx = hRoi.idToIndex(hSf0.uuiduint64);                     % find Scanfield with given uuid in Roi
    hRoi.removeById(idx);                                      % remove Scanfield from Roi

    hRoi.removeByZ(2);                                         % remove Scanfield at z=2 from Roi

Scanfields
==========

Each Roi can contain one or more Scanfields. A Scanfield is a 2D cross-section of the 3D Roi at a particular z level. ScanImage® currently supports multiple types of scanfields.

- RotatedRectangle API - defines an imaging area
- StimulusField API - defines a stimulus path
- IntegrationField API - defines a region, where image data is integrated for online analysis

All fields are defined in the ScanImage® Coordinate Systems coordinate system.

‘imgInfo’ contains the following information:

- number of channels
- number of bytes per pixel
- number of lines per frame
- number of slices
- number of volumes
- number of frames
- the tiff filename
- the ScanImage® version
