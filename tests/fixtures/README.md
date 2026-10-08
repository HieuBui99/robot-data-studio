# OrderPicking v2.1 fixture

`orderpicking-v21/` contains original camera videos and frame data from episodes
0, 10 and 20 of [`ROBOTIS/Task_0002_OrderPicking_lerobot`](https://huggingface.co/datasets/ROBOTIS/Task_0002_OrderPicking_lerobot),
revision `7e54be9075fd31c16309a497489fe25dbc4fe436`. The upstream dataset is
published under Apache-2.0. See [its dataset card](https://huggingface.co/datasets/ROBOTIS/Task_0002_OrderPicking_lerobot/blob/7e54be9075fd31c16309a497489fe25dbc4fe436/README.md).

The three episodes have distinct tasks and all four camera videos. Episode,
task and global frame indices are reindexed to contiguous values; metadata
counts, splits and index statistics are adjusted. Actions, observations,
timestamps, frame indices, and video bytes are preserved. There are 329 frames
at 10 fps, with 19-dimensional actions and states. The two head cameras are
672 × 376; the two wrist cameras are 424 × 240.

The fixture is committed so API tests need neither the full dataset nor Hub
access. Every test project uses the real import endpoint and upstream converter.
