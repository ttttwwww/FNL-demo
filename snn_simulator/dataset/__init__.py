"""
This module is the interface for different type of datasets.
In most situation, simple network will recognize static image as the benchmark.
However, there are various types of data, such as audio, video, etc.
This module provide different function to handle different type of dataset.

The external data should be encapsulated in a custom torch.utils.data.Dataset class.
The reading and management should be implemented by the custom class.
This module is responsible for encoding origin data into spike format
and make it a torch.utils.data.Dataset for simulator
"""

import logging

logger = logging.getLogger(__name__)
