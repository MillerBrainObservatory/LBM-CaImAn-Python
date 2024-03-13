Tips and Tricks for Imaging Data Processing
===========================================

This document, based on insights from Alex Denman (NIH/NIDA), offers comprehensive advice on imaging data processing, including reshaping, motion correction, source extraction, and storage optimization.

Image Reshaping and Inspection
-------------------------------

Visually inspect images post-reshaping to understand how Regions of Interest (ROIs) are rearranged. This is crucial for determining the optimal data organization strategy.

.. list-table:: Visual Inspection After Reshaping
   :widths: 50 50
   :header-rows: 1

   * - Task
     - Description
   * - Reshaping
     - Images are reshaped from [5104 145] to [660 971].
   * - Inspection
     - Essential for understanding ROI rearrangement.

Data Processing Considerations
------------------------------

Identify how different parts of the data (planes, ROIs) are processed together, influencing decisions on data splitting during initial ingestion.

.. list-table:: Data Processing Key Factors
   :widths: 50 50
   :header-rows: 1

   * - Consideration
     - Description
   * - Correlation Check
     - Downsample and check correlations between planes to ensure they match expected patterns, indicating physically neighboring planes ought to be more correlated.

Motion Correction Details
-------------------------

Understand the motion correction process, crucial for data handling and processing strategies.

.. list-table:: Motion Correction Process
   :widths: 30 70
   :header-rows: 1

   * - Option
     - Description
   * - A
     - Each plane is motion-corrected independently.
   * - B
     - The dataset undergoes joint motion correction before saving each plane separately.

ROI Arrangement and Processing
------------------------------

The arrangement and processing of ROIs significantly affect data handling.

.. list-table:: ROI Processing Approaches
   :widths: 30 70
   :header-rows: 1

   * - Approach
     - Description
   * - C
     - ROIs stitched into a cohesive image, requiring reevaluation of ROI configuration.
   * - D
     - ROIs arranged differently within the frame, suggesting a compound image format.

Storage and Data Type Considerations
------------------------------------

Optimize data storage by choosing the appropriate data types and considering compression.

.. list-table:: Storage Optimization Tips
   :widths: 30 70
   :header-rows: 1

   * - Tip
     - Description
   * - Data Type
     - Prefer storing data as `uint16` over `float32` for efficiency, converting to `float32` only when necessary for processing.
   * - Chunk Size
     - For H5 storage, keep chunk size under 1MB to balance read/write speed and compression efficiency.

Motion Correction Optimization
------------------------------

Strategies for efficient motion correction, including storing correction vectors and minimizing data size.

.. list-table:: Motion Correction Efficiency
   :widths: 30 70
   :header-rows: 1

   * - Strategy
     - Description
   * - Vectors Storage
     - Store motion correction vectors compactly, significantly reducing storage compared to full frames.
   * - Fixed Point
     - Utilize fixed point representation for shift vectors to save space without losing precision.

Operational Considerations
--------------------------

Considerations for running source extraction and motion correction to optimize efficiency and convenience.

.. list-table:: Operational Efficiency Tips
   :widths: 30 70
   :header-rows: 1

   * - Consideration
     - Description
   * - Batch Processing
     - Assess whether batch processing ROIs together is for convenience or a necessity, impacting the choice of running ROIs through processes individually.

